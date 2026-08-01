"""Qdrant collection, idempotent ingestion and operational safety.

Closes RF-07 … RF-10, RF-14 and RF-18, including point 1 of the delivery
checklist: re-running the full ingestion must not increase the count.

The integration tests need `make up`. They work on their own collection, never
on the one the delivery uses, so running them cannot destroy a real ingestion.
"""

from __future__ import annotations

import numpy as np
import pytest

from aurum_market.config import RESOURCE_PREFIX, ConfigurationError, Settings
from aurum_market.contracts import CatalogRecord
from aurum_market.ingest import iter_batches, verify_collection
from aurum_market.store.base import CollectionEmptyError, StoreError

from .test_config import make_settings

TEST_COLLECTION = f"{RESOURCE_PREFIX}-tests"


def make_record(index: int, brand: str = "Einhell") -> CatalogRecord:
    return CatalogRecord(
        record_id=f"{index:08x}-a995-56d0-ba03-559885ccef39",
        product_id=f"P{index}",
        title=f"Producto {index}",
        brand=brand,
        color="",
        locale="es",
        text=f"Producto {index}",
        catalog_version=1,
        active=True,
    )


class TestBatching:
    def test_splits_into_deterministic_batches(self) -> None:
        records = [make_record(i) for i in range(10)]
        batches = list(iter_batches(records, batch_size=3))
        assert [offset for offset, _ in batches] == [0, 3, 6, 9]
        assert [len(batch) for _, batch in batches] == [3, 3, 3, 1]

    def test_every_record_appears_exactly_once(self) -> None:
        """Un lote perdido dejaría huecos que solo se verían en las métricas."""
        records = [make_record(i) for i in range(257)]
        seen = [
            r.product_id
            for _, batch in iter_batches(records, batch_size=64)
            for r in batch
        ]
        assert seen == [r.product_id for r in records]

    def test_a_batch_larger_than_the_catalog_yields_one_batch(self) -> None:
        records = [make_record(i) for i in range(5)]
        assert len(list(iter_batches(records, batch_size=999))) == 1

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_an_invalid_batch_size_is_rejected(self, batch_size: int) -> None:
        with pytest.raises(StoreError, match="batch_size"):
            list(iter_batches([make_record(0)], batch_size=batch_size))


class TestHnswSettings:
    def test_indexing_thresholds_are_part_of_the_declared_configuration(self) -> None:
        """Sin bajarlos, Qdrant no construye el índice y RF-08 no sería medible."""
        settings = make_settings()
        declared = settings.hnsw.as_dict()
        assert declared["m"] == 24
        assert declared["ef_construct"] == 120
        assert declared["indexing_threshold"] == 1_000
        assert declared["full_scan_threshold"] == 1_000

    def test_negative_thresholds_are_rejected(self) -> None:
        from aurum_market.config import HnswSettings

        with pytest.raises(ConfigurationError, match="INDEXING_THRESHOLD"):
            HnswSettings(indexing_threshold=-1)


@pytest.mark.integration
class TestQdrantCollection:
    """Needs `make up`. Operates on its own collection."""

    @pytest.fixture
    def store(self):
        from aurum_market.store.qdrant_store import QdrantStore

        settings = make_settings(
            qdrant_collection=TEST_COLLECTION,
            allow_reset=True,
            confirm_cleanup=TEST_COLLECTION,
        )
        store = QdrantStore(settings)
        try:
            store.health()
        except Exception as error:
            pytest.skip(f"Qdrant no disponible: {error}")
        store.reset()
        yield store
        store.reset()

    def test_collection_schema_is_explicit(self, store) -> None:
        store.ensure_collection(dimension=8)
        status = store.status()
        assert status.exists
        assert status.dimension == 8
        assert status.distance.lower() == "cosine"

    def test_the_declared_hnsw_configuration_is_applied(self, store) -> None:
        """RF-08: declarar una configuración y verificar que el motor la aplicó
        son cosas distintas. Esto lee de vuelta lo que Qdrant tiene puesto."""
        store.ensure_collection(dimension=8)
        status = store.status()
        assert status.hnsw_m == 24
        assert status.hnsw_ef_construct == 120
        assert status.indexing_threshold == 1_000
        assert status.full_scan_threshold == 1_000

    @pytest.mark.slow
    def test_the_index_is_actually_built(self, store) -> None:
        """Qdrant responde por fuerza bruta mientras el segmento sea pequeño.

        El umbral está en KILOBYTES y se aplica **por segmento**, no por
        colección: 1.500 vectores de 768 dimensiones ocupan 4.500 KB que, con
        tres segmentos, son 1.500 KB cada uno y superan el límite de 1.000.

        Si este test fallara, m y ef_construct no tendrían efecto observable y
        la fidelidad ANN sería trivialmente 1,0: estaríamos midiendo un índice
        que no se usa.
        """
        dimension = 768
        count = 1_500
        store.ensure_collection(dimension=dimension)
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(count, dimension)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        store.upsert_batch([make_record(i) for i in range(count)], vectors)

        status = store.wait_until_indexed(
            expected_points=count, require_indexed=True, timeout_seconds=180.0
        )
        assert status.indexed_vectors_count >= count
        assert status.fully_indexed
        assert status.kilobytes_per_segment > status.indexing_threshold

    def test_green_status_does_not_mean_indexed(self, store) -> None:
        """El matiz que obliga a declarar require_indexed explícitamente.

        Qdrant almacena y responde `green` de inmediato, y construye el grafo
        después. Esperar solo a `green` demuestra que el dato está, no que el
        índice exista, y una latencia medida antes del grafo mide un escaneo.
        """
        store.ensure_collection(dimension=4)
        records = [make_record(i) for i in range(20)]
        vectors = np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (20, 1))
        store.upsert_batch(records, vectors)

        status = store.wait_until_indexed(expected_points=20)
        assert status.status == "green"
        assert status.points_count == 20
        # 20 vectores de 4 dimensiones son 0,3 KB: muy por debajo del umbral.
        assert status.indexed_vectors_count == 0

    def test_double_ingest_keeps_count(self, store) -> None:
        """Punto 1 de "Antes de entregar", el más importante de la ingesta."""
        store.ensure_collection(dimension=4)
        records = [make_record(i) for i in range(50)]
        vectors = np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (50, 1))

        store.upsert_batch(records, vectors)
        first = store.wait_until_indexed(expected_points=50).points_count

        store.upsert_batch(records, vectors)
        second = store.wait_until_indexed(expected_points=50).points_count

        assert first == second == 50

    def test_upsert_updates_in_place(self, store) -> None:
        """La idempotencia es por diseño: el mismo ID sobrescribe."""
        store.ensure_collection(dimension=4)
        vector = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
        store.upsert_batch([make_record(1, brand="Einhell")], vector)
        store.upsert_batch([make_record(1, brand="Bosch")], vector)

        assert store.status().points_count == 1
        payload = store.get_by_record_id(make_record(1).record_id)
        assert payload is not None
        assert payload["brand"] == "Bosch"

    def test_misaligned_records_and_vectors_are_rejected(self, store) -> None:
        store.ensure_collection(dimension=4)
        with pytest.raises(StoreError, match="Desalineación"):
            store.upsert_batch(
                [make_record(0), make_record(1)],
                np.array([[1.0, 0, 0, 0]], dtype=np.float32),
            )

    def test_search_returns_the_declared_contract(self, store) -> None:
        store.ensure_collection(dimension=4)
        records = [make_record(i) for i in range(4)]
        store.upsert_batch(records, np.eye(4, dtype=np.float32))
        store.wait_until_indexed(expected_points=4)

        hits = store.search_vector(np.array([0, 1.0, 0, 0], dtype=np.float32), top_k=2)
        assert [hit.rank for hit in hits] == [1, 2]
        assert hits[0].product_id == "P1"
        # Qdrant con COSINE devuelve una similitud, no una distancia (P-03).
        assert hits[0].score_kind == "similarity"
        assert hits[0].higher_is_better is True

    def test_filtered_search_never_leaks_another_brand(self, store) -> None:
        """RF-14: el filtro viaja dentro de la consulta, con índice de payload."""
        store.ensure_collection(dimension=4)
        records = [
            make_record(i, brand="Einhell" if i % 2 == 0 else "Bosch") for i in range(8)
        ]
        vectors = np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (8, 1))
        store.upsert_batch(records, vectors)
        store.wait_until_indexed(expected_points=8)

        hits = store.search_vector(
            np.array([1.0, 0, 0, 0], dtype=np.float32), top_k=10, brand="Bosch"
        )
        assert hits
        assert all(hit.brand == "Bosch" for hit in hits)

    def test_a_brand_with_no_products_returns_nothing(self, store) -> None:
        store.ensure_collection(dimension=4)
        store.upsert_batch(
            [make_record(0)], np.array([[1.0, 0, 0, 0]], dtype=np.float32)
        )
        store.wait_until_indexed(expected_points=1)
        hits = store.search_vector(
            np.array([1.0, 0, 0, 0], dtype=np.float32), brand="Inexistente"
        )
        assert hits == []

    def test_searching_a_missing_collection_fails_loudly(self, store) -> None:
        """Una lista vacía en silencio ocultaría una ingesta que no ocurrió."""
        with pytest.raises(CollectionEmptyError, match="no existe"):
            store.search_vector(np.array([1.0, 0, 0, 0], dtype=np.float32))

    def test_searching_an_empty_collection_fails_loudly(self, store) -> None:
        store.ensure_collection(dimension=4)
        with pytest.raises(CollectionEmptyError, match="vacía"):
            store.search_vector(np.array([1.0, 0, 0, 0], dtype=np.float32))

    def test_deleting_removes_the_point(self, store) -> None:
        store.ensure_collection(dimension=4)
        record = make_record(0)
        store.upsert_batch([record], np.array([[1.0, 0, 0, 0]], dtype=np.float32))
        store.wait_until_indexed(expected_points=1)

        store.delete_by_record_ids([record.record_id])
        assert store.get_by_record_id(record.record_id) is None
        assert store.status().points_count == 0

    def test_a_dimension_change_demands_an_explicit_reset(self, store) -> None:
        """La dimensión es parte del esquema, no un parámetro ajustable."""
        store.ensure_collection(dimension=4)
        with pytest.raises(StoreError, match="dimensión"):
            store.ensure_collection(dimension=8)

    def test_ensure_collection_is_idempotent(self, store) -> None:
        assert store.ensure_collection(dimension=4) is True
        assert store.ensure_collection(dimension=4) is False

    def test_verification_catches_a_wrong_count(self, store) -> None:
        store.ensure_collection(dimension=4)
        store.upsert_batch(
            [make_record(0)], np.array([[1.0, 0, 0, 0]], dtype=np.float32)
        )
        store.wait_until_indexed(expected_points=1)
        with pytest.raises(StoreError, match="se esperaban"):
            verify_collection(store, expected_points=99)

    def test_verification_catches_a_wrong_dimension(self, store) -> None:
        store.ensure_collection(dimension=4)
        store.upsert_batch(
            [make_record(0)], np.array([[1.0, 0, 0, 0]], dtype=np.float32)
        )
        store.wait_until_indexed(expected_points=1)
        with pytest.raises(StoreError, match="otro modelo"):
            verify_collection(store, expected_points=1, expected_dimension=768)

    def test_waiting_gives_up_with_an_explanation(self, store) -> None:
        """P-10: si la escritura no se vuelve observable, el sistema lo dice."""
        store.ensure_collection(dimension=4)
        with pytest.raises(StoreError, match="no se ha vuelto observable"):
            store.wait_until_indexed(expected_points=10, timeout_seconds=1.0)


@pytest.mark.integration
class TestOperationalSafety:
    """RF-18: nothing destructive happens without double authorisation."""

    def build(self, **overrides) -> Settings:
        return make_settings(qdrant_collection=TEST_COLLECTION, **overrides)

    def store_for(self, settings):
        from aurum_market.store.qdrant_store import QdrantStore

        store = QdrantStore(settings)
        try:
            store.health()
        except Exception as error:
            pytest.skip(f"Qdrant no disponible: {error}")
        return store

    def test_reset_is_blocked_by_default(self) -> None:
        store = self.store_for(self.build())
        with pytest.raises(StoreError, match="bloqueada"):
            store.reset()

    def test_permission_alone_does_not_authorize(self) -> None:
        store = self.store_for(self.build(allow_reset=True, confirm_cleanup=""))
        with pytest.raises(StoreError, match="bloqueada"):
            store.reset()

    def test_confirming_another_collection_does_not_authorize(self) -> None:
        store = self.store_for(
            self.build(allow_reset=True, confirm_cleanup=f"{RESOURCE_PREFIX}-otra")
        )
        with pytest.raises(StoreError, match="bloqueada"):
            store.reset()

    def test_a_collection_outside_the_prefix_cannot_be_reached(self) -> None:
        """Ningún recurso ajeno a la actividad es alcanzable (P-11)."""
        with pytest.raises(ConfigurationError, match="prefijo protegido"):
            make_settings(qdrant_collection="produccion-catalogo")
