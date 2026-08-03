"""The seven "Antes de entregar" checks, gathered so they run as one gate.

The statement closes with a checklist. Rather than trusting anyone to walk it
manually the night before, each point lives here as an executable test. Those
inspecting the artifacts skip cleanly when they have not been generated yet.
"""

from __future__ import annotations

import json

import pytest

from aurum_market.artifacts import (
    ArtifactError,
    build_duplicate_rows,
    build_search_rows,
    read_duplicate_results,
    read_search_results,
    verify_artifacts,
    write_search_results,
)
from aurum_market.config import ARTIFACTS_DIRECTORY, PROJECT_ROOT, RESULTS_DIRECTORY
from aurum_market.contracts import DuplicateDecision, SearchHit


def make_hit(product_id: str, rank: int = 1, score: float = 0.9) -> SearchHit:
    return SearchHit(
        rank=rank,
        record_id="000bd6e8-a995-56d0-ba03-559885ccef39",
        product_id=product_id,
        title=f"Producto {product_id}",
        brand="Marca",
        native_score=score,
        score_kind="similarity",
        higher_is_better=True,
    )


requires_artifacts = pytest.mark.skipif(
    not (RESULTS_DIRECTORY / "resultados_busqueda.csv").is_file(),
    reason="Ejecuta `aurum deliver` para generar los artefactos",
)


class TestSearchArtifactShape:
    """Rules the artifact builder enforces, independent of any live run."""

    def test_ranks_are_rederived_from_position(self) -> None:
        """Un almacén que devolviera un rank raro no puede colarlo al fichero."""
        hits = [make_hit("P1", rank=7), make_hit("P2", rank=3)]
        rows = build_search_rows({"EVAL-1-direct": hits})
        assert [row["rank"] for row in rows] == [1, 2]

    def test_a_repeated_product_within_one_query_is_rejected(self) -> None:
        """Punto 4: diez IDs ÚNICOS por consulta."""
        with pytest.raises(ArtifactError, match="repite el producto"):
            build_search_rows({"EVAL-1-direct": [make_hit("P1"), make_hit("P1", 2)]})

    def test_the_same_product_may_appear_in_different_queries(self) -> None:
        """La unicidad es por consulta, no global: dos consultas pueden coincidir."""
        rows = build_search_rows(
            {"EVAL-1-direct": [make_hit("P1")], "EVAL-2-direct": [make_hit("P1")]}
        )
        assert len(rows) == 2

    def test_the_native_score_travels_untransformed(self) -> None:
        """P-03: el score llega al fichero con su valor, sin reescalar."""
        rows = build_search_rows({"EVAL-1-direct": [make_hit("P1", score=0.876241)]})
        assert rows[0]["score"] == pytest.approx(0.876241)

    def test_an_invalid_artifact_is_never_written(self, tmp_path) -> None:
        """Validar antes de escribir: un fichero inválido no llega a existir."""
        with pytest.raises(ArtifactError):
            write_search_results(
                {"consulta-con-formato-malo": [make_hit("P1")]}, directory=tmp_path
            )
        assert not (tmp_path / "resultados_busqueda.csv").exists()


class TestDuplicateArtifactShape:
    def test_a_positive_carries_its_candidate(self) -> None:
        rows = build_duplicate_rows(
            [DuplicateDecision("EVAL-DUP-001", True, "B000G3T55M", 0.95)]
        )
        assert rows[0]["matched_product_id"] == "B000G3T55M"

    def test_a_negative_leaves_the_candidate_empty(self) -> None:
        rows = build_duplicate_rows(
            [DuplicateDecision("EVAL-NEW-001", False, "", 0.85)]
        )
        assert rows[0]["matched_product_id"] == ""

    def test_rows_come_out_sorted(self) -> None:
        """Un orden estable hace que dos ejecuciones den ficheros idénticos."""
        rows = build_duplicate_rows(
            [
                DuplicateDecision("EVAL-NEW-001", False, "", 0.85),
                DuplicateDecision("EVAL-DUP-001", True, "B1", 0.95),
            ]
        )
        assert [row["incoming_id"] for row in rows] == ["EVAL-DUP-001", "EVAL-NEW-001"]


@requires_artifacts
class TestDeliveryChecklist:
    """Los siete puntos, contra los artefactos realmente escritos en disco."""

    def test_blind_rankings_have_ten_unique_valid_ids(self) -> None:
        """Punto 4 de "Antes de entregar"."""
        by_query: dict[str, list[str]] = {}
        for row in read_search_results():
            by_query.setdefault(row["evaluation_id"], []).append(row["product_id"])

        assert len(by_query) == 12, "se esperaban las 12 consultas ciegas"
        for query_id, product_ids in by_query.items():
            assert len(product_ids) == 10, f"{query_id} no trae diez resultados"
            assert len(set(product_ids)) == 10, f"{query_id} repite algún producto"
            assert all(product_ids), f"{query_id} trae un product_id vacío"

    def test_ranks_are_consecutive_from_one_to_ten(self) -> None:
        by_query: dict[str, list[int]] = {}
        for row in read_search_results():
            by_query.setdefault(row["evaluation_id"], []).append(row["rank"])
        for query_id, ranks in by_query.items():
            assert sorted(ranks) == list(range(1, 11)), f"{query_id} tiene rangos raros"

    def test_positive_prediction_names_a_candidate(self) -> None:
        """Punto 5 de "Antes de entregar"."""
        rows = read_duplicate_results()
        assert len(rows) == 14
        for row in rows:
            if row["predicted_duplicate"]:
                assert row["matched_product_id"], (
                    f"{row['incoming_id']} predice duplicado sin señalar producto"
                )
            else:
                assert row["matched_product_id"] == ""

    @pytest.mark.slow
    def test_every_matched_product_exists_in_the_catalog(self) -> None:
        """Señalar un producto inexistente sería peor que no señalar ninguno."""
        from aurum_market.data import load_catalog

        catalog = load_catalog("full")
        for row in read_duplicate_results():
            if row["matched_product_id"]:
                assert row["matched_product_id"] in catalog.by_product_id

    def test_the_metrics_artifact_carries_the_required_fields(self) -> None:
        payload = json.loads(
            (RESULTS_DIRECTORY / "metricas_desarrollo.json").read_text(encoding="utf-8")
        )
        for field in (
            "ndcg_at_10",
            "recall_at_10",
            "mrr_at_10",
            "latency_p50_ms",
            "latency_p95_ms",
        ):
            assert field in payload, f"falta {field}"

    def test_latency_ships_with_the_conditions_that_produced_it(self) -> None:
        """RF-21: una latencia sin su entorno no es reproducible ni comparable."""
        payload = json.loads(
            (RESULTS_DIRECTORY / "metricas_desarrollo.json").read_text(encoding="utf-8")
        )
        environment = payload["environment"]
        assert environment["repetitions"] >= 1
        assert "warmup_repetitions" in environment
        assert environment["platform"]

    def test_all_three_artifacts_validate_against_their_contracts(self) -> None:
        """Se validan al escribir y al leer: el ida y vuelta por CSV también cuenta."""
        assert len(verify_artifacts()) == 3

    def test_the_frozen_threshold_is_the_one_that_was_used(self) -> None:
        """Si difirieran, los resultados no vendrían de la configuración declarada."""
        import yaml

        final = yaml.safe_load(
            (PROJECT_ROOT / "config" / "final.yaml").read_text(encoding="utf-8")
        )
        payload = json.loads(
            (RESULTS_DIRECTORY / "metricas_desarrollo.json").read_text(encoding="utf-8")
        )
        assert payload["duplicates"]["threshold"] == final["duplicates"]["threshold"]


class TestSingleEntryPoint:
    def test_deliver_is_the_single_entry_point(self) -> None:
        """Punto 6: las métricas se regeneran desde un único comando."""
        from aurum_market.cli import app

        names = {
            command.name or command.callback.__name__
            for command in app.registered_commands
        }
        assert "deliver" in names

    def test_deliver_never_applies_the_catalog_events(self) -> None:
        """ADR-001: los eventos van después, o destruirían el estado entregado."""
        import inspect

        from aurum_market import cli

        source = inspect.getsource(cli.deliver)
        assert "load_catalog_events" not in source
        assert "apply_events" not in source


@requires_artifacts
class TestErrorAttribution:
    def payload(self) -> dict:
        return json.loads(
            (ARTIFACTS_DIRECTORY / "attribution.json").read_text(encoding="utf-8")
        )

    def test_at_least_three_failures_are_attributed(self) -> None:
        """RF-24 exige al menos tres fallos representativos."""
        assert self.payload()["analysed"] >= 3

    def test_more_than_one_layer_is_represented(self) -> None:
        """Atribuir todo a una capa no demuestra saber diferenciarlas."""
        assert len(self.payload()["by_layer"]) >= 2

    def test_every_attribution_carries_its_evidence(self) -> None:
        for item in self.payload()["attributions"]:
            assert item["evidence"], f"{item['query_id']} no aporta evidencia"
            assert len(item["explanation"]) > 60

    def test_the_four_layers_are_documented(self) -> None:
        """Las cuatro que nombra el enunciado, aunque no todas se hayan dado."""
        layers = self.payload()["layers"]
        assert set(layers) == {
            "representacion",
            "indice",
            "datos_o_filtros",
            "persistencia",
        }


class TestRepositoryHygiene:
    """Punto 7, comprobable sin necesidad de artefactos."""

    def test_no_reserved_data_is_present(self) -> None:
        assert not (PROJECT_ROOT / "datos" / "profesorado").exists()

    def test_env_is_not_tracked(self) -> None:
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert ".env" in gitignore

    def test_no_engine_volume_is_committed(self) -> None:
        assert not (PROJECT_ROOT / "qdrant_storage").exists()

    def test_the_artifacts_directory_is_not_tracked(self) -> None:
        """Es regenerable y pesada; los resultados de entrega sí se versionan."""
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert ".artifacts/" in gitignore
