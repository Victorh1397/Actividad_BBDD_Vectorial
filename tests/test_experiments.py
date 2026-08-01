"""The experiment matrix and the common retrieval interface (RF-01, RF-06)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from aurum_market.config import CONTRACTS_DIRECTORY
from aurum_market.contracts import CatalogRecord, RetrievalQuery
from aurum_market.embeddings import EmbeddingMatrix
from aurum_market.experiments import (
    ExperimentConfig,
    build_matrix,
    build_retriever,
    comparison_table,
    run_experiment,
    write_result,
)
from aurum_market.search import DenseRetriever, Retriever
from aurum_market.store.base import StoreError
from aurum_market.store.exact_store import ExactVectorStore


class FakeEncoder:
    """Deterministic stand-in: one axis per known text, no model download."""

    def __init__(self, vocabulary: list[str], model_id: str = "fake/e5-test") -> None:
        self.model_id = model_id
        self._vocabulary = vocabulary

    @property
    def expected_dimension(self) -> int:
        return len(self._vocabulary)

    def encode(self, texts, *, role, use_cache=True, show_progress=False):
        vectors = np.zeros((len(texts), len(self._vocabulary)), dtype=np.float32)
        for row, text in enumerate(texts):
            for column, term in enumerate(self._vocabulary):
                if term in text.lower():
                    vectors[row, column] = 1.0
            norm = np.linalg.norm(vectors[row])
            vectors[row] = vectors[row] / norm if norm else vectors[row]
        # Un vector nulo no es normalizable: se ancla al primer eje.
        for row in range(len(texts)):
            if not vectors[row].any():
                vectors[row, 0] = 1.0
        return EmbeddingMatrix(
            vectors=vectors,
            model_id=self.model_id,
            role=role,
            normalized=True,
            generated_at="",
            text_sha256="",
        )


def make_record(index: int, title: str, brand: str = "") -> CatalogRecord:
    return CatalogRecord(
        record_id=f"{index:08x}-a995-56d0-ba03-559885ccef39",
        product_id=f"P{index}",
        title=title,
        brand=brand,
        color="",
        locale="es",
        text=title,
        catalog_version=1,
        active=True,
    )


@pytest.fixture
def tiny_catalog():
    return [
        make_record(0, "Taladro inalámbrico 24V", "Einhell"),
        make_record(1, "Vestido largo de fiesta", "Zara"),
        make_record(2, "Silla ergonómica de oficina", "Ikea"),
    ]


class TestDenseRetriever:
    def test_retrieves_the_semantically_closest_product(self, tiny_catalog) -> None:
        encoder = FakeEncoder(["taladro", "vestido", "silla"])
        matrix = encoder.encode([r.title for r in tiny_catalog], role="document")
        retriever = DenseRetriever(
            ExactVectorStore(tiny_catalog, matrix.vectors), encoder
        )
        hits = retriever.search("taladro", top_k=1)
        assert hits[0].product_id == "P0"

    def test_satisfies_the_retriever_protocol(self, tiny_catalog) -> None:
        """TF-IDF y denso deben ser intercambiables para la evaluación (RF-01)."""
        encoder = FakeEncoder(["taladro", "vestido", "silla"])
        matrix = encoder.encode([r.title for r in tiny_catalog], role="document")
        retriever = DenseRetriever(
            ExactVectorStore(tiny_catalog, matrix.vectors), encoder
        )
        assert isinstance(retriever, Retriever)

    def test_an_empty_query_is_rejected(self, tiny_catalog) -> None:
        encoder = FakeEncoder(["taladro"])
        matrix = encoder.encode([r.title for r in tiny_catalog], role="document")
        retriever = DenseRetriever(
            ExactVectorStore(tiny_catalog, matrix.vectors), encoder
        )
        with pytest.raises(StoreError, match="vacía"):
            retriever.search("  ")

    def test_a_store_built_with_another_model_is_rejected(self, tiny_catalog) -> None:
        """Mezclar dimensiones daría resultados sin sentido, no un error obvio."""
        store = ExactVectorStore(tiny_catalog, np.eye(3, 5, dtype=np.float32))
        with pytest.raises(StoreError, match="otro modelo"):
            DenseRetriever(store, FakeEncoder(["a", "b", "c"]))

    def test_the_brand_filter_reaches_the_store(self, tiny_catalog) -> None:
        encoder = FakeEncoder(["taladro", "vestido", "silla"])
        matrix = encoder.encode([r.title for r in tiny_catalog], role="document")
        retriever = DenseRetriever(
            ExactVectorStore(tiny_catalog, matrix.vectors), encoder
        )
        hits = retriever.search("taladro", top_k=5, brand="Zara")
        assert all(hit.brand == "Zara" for hit in hits)


class TestMatrixDesign:
    def test_declares_the_four_planned_runs(self) -> None:
        matrix = build_matrix()
        assert [c.experiment_id for c in matrix] == ["E0", "E1", "E2", "E3"]

    def test_every_run_states_what_it_isolates(self) -> None:
        """Cambiar el modelo sin analizar no es un experimento (RF-06)."""
        for config in build_matrix():
            assert config.isolates
            assert len(config.description) > 30

    def test_each_run_changes_one_variable_from_its_comparison(self) -> None:
        matrix = {c.experiment_id: c for c in build_matrix()}
        # E1 vs E0: mismo texto, cambia el recuperador.
        assert matrix["E1"].text_strategy == matrix["E0"].text_strategy
        assert matrix["E1"].retriever != matrix["E0"].retriever
        # E2 vs E1: mismo modelo, cambia el texto.
        assert matrix["E2"].embedding_model == matrix["E1"].embedding_model
        assert matrix["E2"].text_strategy != matrix["E1"].text_strategy
        # E3 vs E2: mismo texto, cambia el modelo.
        assert matrix["E3"].embedding_model != matrix["E2"].embedding_model

    def test_e3_inherits_the_winning_strategy(self) -> None:
        """Su única variable debe ser el tamaño del modelo."""
        assert build_matrix(winner_strategy="raw_text")[3].text_strategy == "raw_text"
        assert (
            build_matrix(winner_strategy="title_brand_color")[3].text_strategy
            == "title_brand_color"
        )

    def test_the_baseline_needs_no_model(self) -> None:
        assert build_matrix()[0].embedding_model is None


class TestRunningAnExperiment:
    def build_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="E9",
            retriever="tfidf",
            text_strategy="title_only",
            isolates="una comprobación",
        )

    def test_produces_a_result_that_matches_its_contract(
        self, tiny_catalog, tmp_path
    ) -> None:
        queries = [RetrievalQuery(query_id="Q1", text="taladro inalámbrico")]
        judgments = {"Q1": {"P0": 3.0, "P1": 0.0}}
        result = run_experiment(
            self.build_config(), tiny_catalog, queries, judgments, top_k=2
        )
        path = write_result(result, directory=tmp_path)

        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = json.loads(
            (CONTRACTS_DIRECTORY / "experiment_run.schema.json").read_text(
                encoding="utf-8"
            )
        )
        from jsonschema import Draft202012Validator

        Draft202012Validator(schema).validate(payload)

    def test_records_configuration_metrics_and_ids(self, tiny_catalog) -> None:
        """P-06: sin los tres, un número es una anécdota."""
        queries = [RetrievalQuery(query_id="Q1", text="taladro")]
        result = run_experiment(
            self.build_config(), tiny_catalog, queries, {"Q1": {"P0": 3.0}}, top_k=2
        )
        payload = result.as_dict()
        assert payload["configuration"]["text_strategy"] == "title_only"
        assert payload["configuration"]["top_k"] == 2
        # La profundidad se declara una vez, en configuration, no en el nombre
        # de cada métrica.
        assert set(payload["metrics"]) >= {"ndcg", "recall", "mrr"}
        assert payload["retrieved_ids"]["Q1"]

    def test_finds_the_relevant_product(self, tiny_catalog) -> None:
        queries = [RetrievalQuery(query_id="Q1", text="taladro inalámbrico")]
        result = run_experiment(
            self.build_config(), tiny_catalog, queries, {"Q1": {"P0": 3.0}}, top_k=1
        )
        assert result.retrieved_ids["Q1"] == ["P0"]
        assert result.report.mean_ndcg == pytest.approx(1.0)

    def test_notes_warn_that_sample_figures_are_optimistic(self, tiny_catalog) -> None:
        queries = [RetrievalQuery(query_id="Q1", text="taladro")]
        result = run_experiment(
            self.build_config(), tiny_catalog, queries, {"Q1": {"P0": 3.0}}, top_k=1
        )
        assert any("distractores" in note for note in result.notes)

    def test_an_unknown_retriever_is_rejected(self, tiny_catalog) -> None:
        config = ExperimentConfig(
            experiment_id="E9", retriever="magia", text_strategy="title_only"
        )
        with pytest.raises(ValueError, match="Recuperador desconocido"):
            build_retriever(config, tiny_catalog)

    def test_a_dense_run_without_a_model_is_rejected(self, tiny_catalog) -> None:
        config = ExperimentConfig(
            experiment_id="E9", retriever="dense_exact", text_strategy="title_only"
        )
        with pytest.raises(ValueError, match="declarar un modelo"):
            build_retriever(config, tiny_catalog)


class TestComparisonTable:
    def test_renders_one_row_per_experiment_with_the_ceiling(
        self, tiny_catalog
    ) -> None:
        queries = [RetrievalQuery(query_id="Q1", text="taladro")]
        judgments = {"Q1": {"P0": 3.0, "P1": 2.0}}
        results = [
            run_experiment(
                ExperimentConfig(
                    experiment_id=f"E{i}",
                    retriever="tfidf",
                    text_strategy="title_only",
                ),
                tiny_catalog,
                queries,
                judgments,
                top_k=2,
            )
            for i in range(2)
        ]
        table = comparison_table(results)
        assert "nDCG@10" in table
        assert "E0" in table and "E1" in table
        assert "techo estructural" in table
