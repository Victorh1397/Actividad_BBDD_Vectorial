"""Text composition, exact oracle and lexical baseline (RF-02, RF-03, RF-20)."""

from __future__ import annotations

import numpy as np
import pytest

from aurum_market.baselines import TfidfRetriever
from aurum_market.contracts import CatalogRecord
from aurum_market.data import load_catalog
from aurum_market.store import CollectionEmptyError, ExactVectorStore, StoreError
from aurum_market.text import (
    TEXT_STRATEGIES,
    TextCompositionError,
    compose,
    compose_all,
    describe,
)

VALID_UUID = "000bd6e8-a995-56d0-ba03-559885ccef39"
OTHER_UUID = "0037a9df-8492-508f-8167-c09624801216"


def make_record(**overrides: object) -> CatalogRecord:
    defaults: dict[str, object] = {
        "record_id": VALID_UUID,
        "product_id": "B0818K237B",
        "title": "Taladro inalámbrico 24V",
        "brand": "Einhell",
        "color": "Rojo",
        "locale": "es",
        "text": "Taladro inalámbrico 24V. Marca: Einhell. Características: dos baterías.",
        "catalog_version": 1,
        "active": True,
    }
    return CatalogRecord(**(defaults | overrides))  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def sample_records():
    return load_catalog("sample").records


class TestTextStrategies:
    def test_raw_text_returns_the_catalog_field(self) -> None:
        record = make_record()
        assert compose(record, "raw_text") == record.text

    def test_title_only_returns_just_the_title(self) -> None:
        assert compose(make_record(), "title_only") == "Taladro inalámbrico 24V"

    def test_title_brand_color_labels_the_metadata(self) -> None:
        composed = compose(make_record(), "title_brand_color")
        assert composed == "Taladro inalámbrico 24V. Marca: Einhell. Color: Rojo"

    def test_absent_metadata_is_skipped_not_rendered_empty(self) -> None:
        """Codificar "Color: " enseñaría al modelo un patrón sin significado."""
        composed = compose(make_record(brand="", color=""), "title_brand_color")
        assert composed == "Taladro inalámbrico 24V"
        assert "Marca:" not in composed
        assert "Color:" not in composed

    def test_partial_metadata_keeps_only_what_exists(self) -> None:
        composed = compose(make_record(color=""), "title_brand_color")
        assert composed == "Taladro inalámbrico 24V. Marca: Einhell"

    def test_raw_text_falls_back_to_the_title(self) -> None:
        assert compose(make_record(text=""), "raw_text") == "Taladro inalámbrico 24V"

    def test_an_unknown_strategy_is_rejected(self) -> None:
        with pytest.raises(TextCompositionError, match="desconocida"):
            compose(make_record(), "titulo_y_precio")  # type: ignore[arg-type]

    def test_every_strategy_is_documented(self) -> None:
        """El enunciado exige documentar qué campos forman el texto."""
        for strategy in TEXT_STRATEGIES:
            assert len(describe(strategy)) > 40

    def test_compose_all_preserves_order(self, sample_records) -> None:
        """Si el orden cambiara, los vectores dejarían de alinearse con los IDs."""
        texts = compose_all(sample_records[:20], "title_only")
        assert len(texts) == 20
        assert texts[0] == sample_records[0].title

    def test_composition_never_produces_empty_text(self, sample_records) -> None:
        for strategy in TEXT_STRATEGIES:
            for text in compose_all(sample_records[:200], strategy):
                assert text.strip()

    def test_raw_text_is_much_longer_than_the_composed_one(
        self, sample_records
    ) -> None:
        """La razón de ser del experimento E1 vs E2 (27,2 % excede 512 tokens)."""
        subset = sample_records[:500]
        raw = np.mean([len(t) for t in compose_all(subset, "raw_text")])
        composed = np.mean([len(t) for t in compose_all(subset, "title_brand_color")])
        assert raw > composed * 5


class TestExactVectorStore:
    def build(self, count: int = 4):
        records = [
            make_record(
                record_id=f"{i:08x}-a995-56d0-ba03-559885ccef39",
                product_id=f"P{i}",
                title=f"Producto {i}",
                brand="Einhell" if i % 2 == 0 else "Bosch",
            )
            for i in range(count)
        ]
        # Vectores unitarios sobre ejes distintos: el vecino es predecible.
        embeddings = np.eye(count, dtype=np.float32)
        return records, embeddings

    def test_finds_the_obvious_nearest_neighbour(self) -> None:
        records, embeddings = self.build()
        store = ExactVectorStore(records, embeddings)
        hits = store.search_vector(embeddings[2], top_k=1)
        assert hits[0].product_id == "P2"
        assert hits[0].native_score == pytest.approx(1.0)

    def test_ranks_are_consecutive_from_one(self) -> None:
        records, embeddings = self.build()
        hits = ExactVectorStore(records, embeddings).search_vector(
            embeddings[0], top_k=3
        )
        assert [hit.rank for hit in hits] == [1, 2, 3]

    def test_score_is_declared_as_a_similarity(self) -> None:
        """Con vectores normalizados, el producto interno ES el coseno (P-03)."""
        records, embeddings = self.build()
        hit = ExactVectorStore(records, embeddings).search_vector(embeddings[0])[0]
        assert hit.score_kind == "similarity"
        assert hit.higher_is_better is True

    def test_brand_filter_never_leaks_another_brand(self) -> None:
        records, embeddings = self.build(6)
        store = ExactVectorStore(records, embeddings)
        hits = store.search_vector(embeddings[1], top_k=10, brand="Einhell")
        assert hits
        assert all(hit.brand == "Einhell" for hit in hits)

    def test_a_brand_with_no_products_returns_an_empty_list(self) -> None:
        """Un filtro sin resultados es una respuesta legítima, no un error."""
        records, embeddings = self.build()
        store = ExactVectorStore(records, embeddings)
        assert store.search_vector(embeddings[0], brand="Inexistente") == []

    def test_top_k_larger_than_the_collection_is_capped(self) -> None:
        records, embeddings = self.build(3)
        hits = ExactVectorStore(records, embeddings).search_vector(
            embeddings[0], top_k=99
        )
        assert len(hits) == 3

    def test_an_empty_collection_raises_instead_of_returning_nothing(self) -> None:
        """Una lista vacía en silencio dejaría pasar una ingesta que no ocurrió."""
        store = ExactVectorStore([], np.zeros((0, 4), dtype=np.float32))
        with pytest.raises(CollectionEmptyError):
            store.search_vector(np.zeros(4, dtype=np.float32))

    def test_misaligned_records_and_vectors_are_rejected(self) -> None:
        records, embeddings = self.build(4)
        with pytest.raises(StoreError, match="Desalineación"):
            ExactVectorStore(records, embeddings[:3])

    def test_unnormalized_embeddings_are_rejected(self) -> None:
        """Sin normalizar, el producto interno deja de ser el coseno."""
        records, _ = self.build(2)
        with pytest.raises(StoreError, match="normalizados"):
            ExactVectorStore(
                records, np.array([[3.0, 0.0], [0.0, 5.0]], dtype=np.float32)
            )

    def test_a_query_of_the_wrong_dimension_is_rejected(self) -> None:
        records, embeddings = self.build()
        store = ExactVectorStore(records, embeddings)
        with pytest.raises(StoreError, match="dimensión"):
            store.search_vector(np.zeros(7, dtype=np.float32))

    def test_ties_are_broken_deterministically(self) -> None:
        """Dos productos idénticos no pueden intercambiarse entre ejecuciones."""
        records, _ = self.build(3)
        identical = np.tile(np.array([1.0, 0.0], dtype=np.float32), (3, 1))
        store = ExactVectorStore(records, identical)
        query = np.array([1.0, 0.0], dtype=np.float32)
        first = [hit.product_id for hit in store.search_vector(query, top_k=3)]
        assert first == ["P0", "P1", "P2"]
        assert first == [hit.product_id for hit in store.search_vector(query, top_k=3)]


class TestTfidfBaseline:
    def test_matches_a_literal_query(self, sample_records) -> None:
        retriever = TfidfRetriever(sample_records[:400], strategy="title_brand_color")
        hits = retriever.search("lámpara de pie regulable LED", top_k=5)
        assert hits
        assert "mpara" in hits[0].title.lower()

    def test_results_follow_the_search_hit_contract(self, sample_records) -> None:
        retriever = TfidfRetriever(sample_records[:200], strategy="title_only")
        hits = retriever.search("vestido", top_k=3)
        assert [hit.rank for hit in hits] == [1, 2, 3]
        assert all(hit.product_id for hit in hits)
        assert all(hit.score_kind == "similarity" for hit in hits)

    def test_scores_are_ordered_descending(self, sample_records) -> None:
        retriever = TfidfRetriever(sample_records[:200], strategy="title_only")
        scores = [
            hit.native_score for hit in retriever.search("vestido mujer", top_k=8)
        ]
        assert scores == sorted(scores, reverse=True)

    def test_accents_do_not_split_the_vocabulary(self, sample_records) -> None:
        """En español, "camara" y "cámara" deben ser el mismo término."""
        retriever = TfidfRetriever(sample_records[:400], strategy="title_only")
        with_accent = [h.product_id for h in retriever.search("lámpara", top_k=5)]
        without = [h.product_id for h in retriever.search("lampara", top_k=5)]
        assert with_accent == without

    def test_brand_filter_never_leaks_another_brand(self, sample_records) -> None:
        records = [r for r in sample_records[:600]]
        brand = next(r.brand for r in records if r.brand)
        retriever = TfidfRetriever(records, strategy="title_brand_color")
        hits = retriever.search("producto", top_k=10, brand=brand)
        assert all(hit.brand == brand for hit in hits)

    def test_an_empty_query_is_rejected(self, sample_records) -> None:
        retriever = TfidfRetriever(sample_records[:50], strategy="title_only")
        with pytest.raises(StoreError, match="vacía"):
            retriever.search("   ")

    def test_an_empty_corpus_is_rejected(self) -> None:
        with pytest.raises(CollectionEmptyError):
            TfidfRetriever([], strategy="title_only")

    def test_a_semantic_query_defeats_lexical_matching(self, sample_records) -> None:
        """El problema de negocio, reproducido: sin palabras en común, no hay match.

        Es justo lo que el sistema denso debe resolver, y la razón de que este
        baseline sea la referencia interpretable que pide el enunciado.
        """
        retriever = TfidfRetriever(sample_records[:800], strategy="title_brand_color")
        hits = retriever.search(
            "necesito algo para que no se me caiga el aparato del salón", top_k=5
        )
        # Cualquier coincidencia es marginal: comparte palabras vacías, no sentido.
        assert all(hit.native_score < 0.35 for hit in hits)
