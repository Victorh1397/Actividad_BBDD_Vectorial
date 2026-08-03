"""Catalog mutations: order, idempotence and visibility (RF-16).

The checklist point these tests close — "los eventos dejan exactamente el estado
esperado" — cannot be checked by counting points. These 24 events add eight
products and remove eight, so the total is 15.000 before and 15.000 after. Every
assertion here is therefore about *identity*: which IDs are present, and with
which sheet.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from aurum_market.contracts import CatalogEvent, CatalogRecord, SearchHit
from aurum_market.events import (
    EventError,
    EventReport,
    apply_events,
    classify,
    snapshot_state,
)

NAMESPACE = uuid.UUID("34ef9344-7a3f-5eb2-b30b-aceff745758d")


def record_id_for(product_id: str) -> str:
    return str(uuid.uuid5(NAMESPACE, product_id))


def make_record(product_id: str, title: str, brand: str = "Marca") -> CatalogRecord:
    return CatalogRecord(
        record_id=record_id_for(product_id),
        product_id=product_id,
        title=title,
        brand=brand,
        color="negro",
        locale="es",
        text=title,
        catalog_version=1,
        active=True,
    )


def make_event(
    sequence: int, operation: str, product_id: str, title: str = ""
) -> CatalogEvent:
    return CatalogEvent(
        sequence=sequence,
        event_id=f"EVT-{sequence:03d}",
        operation=operation,  # type: ignore[arg-type]
        record_id=record_id_for(product_id),
        product_id=product_id,
        record=make_record(product_id, title) if operation == "UPSERT" else None,
    )


class FakeStatus:
    exists = True

    def __init__(self, points_count: int) -> None:
        self.points_count = points_count


class FakeStore:
    """An in-memory stand-in that records the order it was called in.

    Deliberately not a mock: it stores payloads and vectors and answers
    searches by exact cosine, so the visibility probes exercise the same code
    path they will against Qdrant.
    """

    collection = "aurum-market-test"

    def __init__(self) -> None:
        self.points: dict[str, tuple[dict[str, object], np.ndarray]] = {}
        self.calls: list[tuple[str, str]] = []

    # --- superficie que events.py consume -------------------------------
    def status(self) -> FakeStatus:
        return FakeStatus(len(self.points))

    def get_by_record_id(self, record_id: str) -> dict[str, object] | None:
        stored = self.points.get(record_id)
        return dict(stored[0]) if stored else None

    def upsert_batch(self, records, vectors) -> int:
        matrix = np.asarray(vectors, dtype=np.float32)
        for position, record in enumerate(records):
            self.calls.append(("UPSERT", record.record_id))
            self.points[record.record_id] = (record.payload(), matrix[position])
        return len(records)

    def delete_by_record_ids(self, record_ids) -> int:
        ids = list(record_ids)
        for record_id in ids:
            self.calls.append(("DELETE", record_id))
            self.points.pop(record_id, None)
        return len(ids)

    def search_vector(self, query_vector, *, top_k: int = 10, brand=None):
        query = np.asarray(query_vector, dtype=np.float32)
        scored = []
        for record_id, (payload, vector) in self.points.items():
            score = float(np.dot(query, vector))
            scored.append((score, record_id, payload))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchHit(
                rank=position + 1,
                record_id=record_id,
                product_id=str(payload["product_id"]),
                title=str(payload["title"]),
                brand=str(payload["brand"]),
                native_score=score,
                score_kind="similarity",
                higher_is_better=True,
            )
            for position, (score, record_id, payload) in enumerate(scored[:top_k])
        ]

    # --- ayudas para los tests ------------------------------------------
    def seed(self, record: CatalogRecord, vector: np.ndarray) -> None:
        self.points[record.record_id] = (record.payload(), vector)


class FakeEncoder:
    """Maps text to a deterministic unit vector, so ranking is predictable.

    Identical text yields an identical vector, which is what makes "the
    product is the best answer to its own description" hold without needing a
    real model.
    """

    model_id = "fake"
    expected_dimension = 16

    def encode(self, texts, *, role: str = "document", show_progress: bool = False):
        vectors = np.stack([self._vector(text) for text in texts])

        class Matrix:
            def __init__(self, values: np.ndarray) -> None:
                self.vectors = values
                self.dimension = values.shape[1]

        return Matrix(vectors)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        seed = int.from_bytes(text.encode("utf-8")[:8].ljust(8, b"\0"), "little")
        generator = np.random.default_rng(seed % (2**32))
        vector = generator.normal(size=16).astype(np.float32)
        return vector / np.linalg.norm(vector)


@pytest.fixture
def world() -> tuple[FakeStore, FakeEncoder]:
    """A collection holding two products the events will later mutate."""
    store, encoder = FakeStore(), FakeEncoder()
    for product_id, title in (
        ("OLD-1", "Producto antiguo uno"),
        ("OLD-2", "Producto antiguo dos"),
    ):
        record = make_record(product_id, title)
        vector = encoder.encode([f"{title}. Marca: Marca. Color: negro"]).vectors[0]
        store.seed(record, vector)
    return store, encoder


class TestClassification:
    """El fichero dice UPSERT; alta o actualización lo dice la colección."""

    def test_an_upsert_on_an_absent_id_is_an_alta(self, world) -> None:
        store, _ = world
        assert classify(make_event(1, "UPSERT", "NEW-1", "Nuevo"), store) == "alta"

    def test_an_upsert_on_a_present_id_is_an_actualizacion(self, world) -> None:
        store, _ = world
        event = make_event(1, "UPSERT", "OLD-1", "Producto antiguo uno - revisado")
        assert classify(event, store) == "actualizacion"

    def test_a_delete_is_always_a_baja(self, world) -> None:
        """Incluso si no borra nada: la intención del evento no cambia."""
        store, _ = world
        assert classify(make_event(1, "DELETE", "AUSENTE"), store) == "baja"


class TestOrdering:
    def test_events_apply_in_sequence_order(self, world) -> None:
        store, encoder = world
        events = [
            make_event(3, "UPSERT", "NEW-2", "Tercero"),
            make_event(1, "UPSERT", "NEW-1", "Primero"),
            make_event(2, "DELETE", "OLD-1"),
        ]
        apply_events(events, store, encoder, measure_visibility=False)
        assert store.calls == [
            ("UPSERT", record_id_for("NEW-1")),
            ("DELETE", record_id_for("OLD-1")),
            ("UPSERT", record_id_for("NEW-2")),
        ]

    def test_a_gap_in_the_sequence_is_rejected(self, world) -> None:
        """Una secuencia con huecos significa que falta un evento del fichero."""
        store, encoder = world
        events = [
            make_event(1, "UPSERT", "NEW-1", "Uno"),
            make_event(3, "DELETE", "OLD-1"),
        ]
        with pytest.raises(EventError, match="sin huecos"):
            apply_events(events, store, encoder, measure_visibility=False)


class TestIdempotence:
    """Punto 3 de "Antes de entregar"."""

    def events(self) -> list[CatalogEvent]:
        return [
            make_event(1, "UPSERT", "OLD-1", "Producto antiguo uno - ficha revisada"),
            make_event(2, "DELETE", "OLD-2"),
            make_event(3, "UPSERT", "NEW-1", "Producto nuevo"),
        ]

    def test_events_are_idempotent(self, world) -> None:
        """El estado tras dos aplicaciones es idéntico al de una."""
        store, encoder = world
        events = self.events()

        apply_events(events, store, encoder, measure_visibility=False)
        after_first = snapshot_state(events, store)

        apply_events(events, store, encoder, measure_visibility=False)
        after_second = snapshot_state(events, store)

        assert after_first == after_second
        assert after_first == {
            record_id_for("OLD-1"): "Producto antiguo uno - ficha revisada",
            record_id_for("OLD-2"): None,
            record_id_for("NEW-1"): "Producto nuevo",
        }

    def test_the_second_run_reports_no_alta_and_no_effective_baja(self, world) -> None:
        """La reaplicación cambia lo que se *informa*, no lo que se *tiene*."""
        store, encoder = world
        events = self.events()

        first = apply_events(events, store, encoder, measure_visibility=False)
        second = apply_events(events, store, encoder, measure_visibility=False)

        assert (first.altas, first.actualizaciones, first.bajas_efectivas) == (1, 1, 1)
        assert (second.altas, second.actualizaciones, second.bajas_efectivas) == (
            0,
            2,
            0,
        )
        assert first.points_after == second.points_after

    def test_a_deletion_of_an_absent_point_is_not_counted_as_effective(
        self, world
    ) -> None:
        store, encoder = world
        report = apply_events(
            [make_event(1, "DELETE", "NUNCA-EXISTIO")],
            store,
            encoder,
            measure_visibility=False,
        )
        assert report.bajas == 1
        assert report.bajas_efectivas == 0
        assert report.expected_points == report.points_before


class TestExpectedCount:
    def test_the_count_alone_cannot_detect_a_run_that_did_nothing(self, world) -> None:
        """Por qué el estado se comprueba por identidad y no por recuento.

        Una alta y una baja dejan el total intacto: 2 puntos antes, 2 después.
        """
        store, encoder = world
        report = apply_events(
            [
                make_event(1, "UPSERT", "NEW-1", "Nuevo"),
                make_event(2, "DELETE", "OLD-1"),
            ],
            store,
            encoder,
            measure_visibility=False,
        )
        assert report.points_before == report.points_after == 2
        assert report.altas == 1 and report.bajas_efectivas == 1
        assert snapshot_state([make_event(1, "DELETE", "OLD-1")], store) == {
            record_id_for("OLD-1"): None
        }

    def test_a_collection_stuck_at_the_old_count_is_reported(self, world) -> None:
        """La espera es exacta: un recuento mayor no satisface la comprobación."""
        store, encoder = world

        class FrozenStore(FakeStore):
            def delete_by_record_ids(self, record_ids) -> int:
                return len(list(record_ids))  # confirma la baja y no la aplica

        frozen = FrozenStore()
        frozen.points = dict(store.points)
        with pytest.raises(EventError, match="puntos esperados"):
            apply_events(
                [make_event(1, "DELETE", "OLD-1")],
                frozen,
                encoder,
                measure_visibility=False,
                timeout_seconds=1.0,
            )


class TestVisibilityProbes:
    """RF-16: por lectura por ID *y* por consulta vectorial, con espera acotada."""

    def test_one_probe_per_kind_is_produced(self, world) -> None:
        store, encoder = world
        events = [
            make_event(1, "UPSERT", "OLD-1", "Producto antiguo uno - ficha revisada"),
            make_event(2, "DELETE", "OLD-2"),
            make_event(3, "UPSERT", "NEW-1", "Producto nuevo"),
            make_event(4, "UPSERT", "NEW-2", "Otro producto nuevo"),
        ]
        report = apply_events(events, store, encoder)
        assert {probe.kind for probe in report.probes} == {
            "alta",
            "actualizacion",
            "baja",
        }
        assert len(report.probes) == 3, "una sonda por tipo, no una por evento"

    def test_an_update_probe_demands_the_new_sheet_not_mere_presence(
        self, world
    ) -> None:
        """Que el punto esté no prueba nada: ya estaba. Prueba que cambió."""
        store, encoder = world
        report = apply_events(
            [make_event(1, "UPSERT", "OLD-1", "Producto antiguo uno - ficha revisada")],
            store,
            encoder,
        )
        probe = report.probes[0]
        assert probe.kind == "actualizacion"
        assert "ficha revisada" in probe.id_evidence
        assert "ficha nueva" in probe.search_evidence

    def test_a_deletion_probe_records_that_it_was_findable_first(self, world) -> None:
        """Sin la observación previa, "no aparece" no demuestra nada."""
        store, encoder = world
        report = apply_events([make_event(1, "DELETE", "OLD-1")], store, encoder)
        probe = report.probes[0]
        assert probe.baseline.startswith("antes de la baja aparecía en la posición")
        assert "ausente del top" in probe.search_evidence

    def test_both_channels_are_measured(self, world) -> None:
        store, encoder = world
        report = apply_events(
            [make_event(1, "UPSERT", "NEW-1", "Nuevo")], store, encoder
        )
        probe = report.probes[0]
        assert probe.by_id_ms >= 0.0
        assert probe.by_search_ms >= 0.0
        assert probe.id_evidence and probe.search_evidence

    def test_an_unobservable_write_is_reported_not_ignored(self, world) -> None:
        """P-10: esperar, fallar o informar. Nunca seguir como si nada."""
        _, encoder = world

        class SilentStore(FakeStore):
            def upsert_batch(self, records, vectors) -> int:
                return len(records)  # confirma la escritura y no la guarda

        with pytest.raises(EventError, match="lectura por ID"):
            apply_events(
                [make_event(1, "UPSERT", "NEW-1", "Nuevo")],
                SilentStore(),
                encoder,
                timeout_seconds=1.0,
            )


class TestReportShape:
    def test_the_report_serialises_for_the_artifact(self, world) -> None:
        store, encoder = world
        report = apply_events(
            [make_event(1, "UPSERT", "NEW-1", "Nuevo")], store, encoder
        )
        payload = report.as_dict()
        assert payload["applied"] == 1
        assert payload["kinds"] == [{"event_id": "EVT-001", "kind": "alta"}]
        assert payload["probes"][0]["kind"] == "alta"
        assert isinstance(report, EventReport)


@pytest.mark.slow
class TestRealEventFile:
    """Hechos sobre el fichero entregado, sin necesidad de motor."""

    def events(self):
        from aurum_market.data import load_catalog_events

        return load_catalog_events()

    def test_the_file_ships_twenty_four_ordered_events(self) -> None:
        events = self.events()
        assert len(events) == 24
        assert [event.sequence for event in events] == list(range(1, 25))

    def test_no_product_is_both_upserted_and_deleted(self) -> None:
        """Sin solape, el orden no altera el estado final. Conviene saberlo.

        Es lo que hace que la idempotencia aquí sea barata: si un producto se
        borrara y se volviera a dar de alta, el resultado dependería del orden
        y la reaplicación sería mucho más delicada.
        """
        events = self.events()
        upserted = {event.product_id for event in events if not event.is_deletion}
        deleted = {event.product_id for event in events if event.is_deletion}
        assert not (upserted & deleted)

    def test_deletions_carry_no_sheet_and_upserts_do(self) -> None:
        for event in self.events():
            assert (event.record is None) == event.is_deletion
