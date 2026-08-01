"""Embedding contract: E5 prefixes, normalisation and caching (RF-04)."""

from __future__ import annotations

import numpy as np
import pytest

from aurum_market.embeddings import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    EmbeddingError,
    EmbeddingMatrix,
    Encoder,
    apply_prefix,
    is_e5_model,
    texts_digest,
)


class TestModelFamilyDetection:
    @pytest.mark.parametrize(
        "model_id",
        [
            "intfloat/multilingual-e5-small",
            "intfloat/multilingual-e5-base",
            "intfloat/e5-large-v2",
            "E5-Small",
        ],
    )
    def test_recognises_the_e5_family(self, model_id: str) -> None:
        assert is_e5_model(model_id) is True

    @pytest.mark.parametrize(
        "model_id",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "BAAI/bge-small-en",
            "hiiamsid/sentence_similarity_spanish_es",
        ],
    )
    def test_other_families_are_not_e5(self, model_id: str) -> None:
        assert is_e5_model(model_id) is False

    def test_an_empty_model_id_is_rejected(self) -> None:
        with pytest.raises(EmbeddingError):
            is_e5_model("   ")


class TestPrefixes:
    MODEL = "intfloat/multilingual-e5-small"

    def test_queries_get_the_query_prefix(self) -> None:
        result = apply_prefix(["taladro 24v"], model_id=self.MODEL, role="query")
        assert result == (f"{QUERY_PREFIX}taladro 24v",)

    def test_documents_get_the_passage_prefix(self) -> None:
        result = apply_prefix(["Taladro Einhell"], model_id=self.MODEL, role="document")
        assert result == (f"{DOCUMENT_PREFIX}Taladro Einhell",)

    def test_non_e5_models_receive_the_original_text(self) -> None:
        """Un prefijo es el contrato de una familia de modelos, no contenido."""
        result = apply_prefix(
            ["taladro 24v"], model_id="BAAI/bge-small-en", role="query"
        )
        assert result == ("taladro 24v",)

    def test_applying_twice_does_not_duplicate_the_prefix(self) -> None:
        once = apply_prefix(["taladro"], model_id=self.MODEL, role="query")
        twice = apply_prefix(once, model_id=self.MODEL, role="query")
        assert once == twice

    def test_query_and_document_prefixes_differ(self) -> None:
        """Codificar un documento como consulta degrada la búsqueda en silencio."""
        as_query = apply_prefix(["taladro"], model_id=self.MODEL, role="query")
        as_document = apply_prefix(["taladro"], model_id=self.MODEL, role="document")
        assert as_query != as_document

    def test_an_unknown_role_is_rejected(self) -> None:
        with pytest.raises(EmbeddingError, match="rol"):
            apply_prefix(["x"], model_id=self.MODEL, role="titulo")  # type: ignore[arg-type]

    def test_empty_texts_are_rejected(self) -> None:
        with pytest.raises(EmbeddingError, match="no vacías"):
            apply_prefix(["   "], model_id=self.MODEL, role="query")

    def test_a_bare_string_is_not_a_sequence_of_texts(self) -> None:
        with pytest.raises(EmbeddingError, match="secuencia"):
            apply_prefix("taladro", model_id=self.MODEL, role="query")  # type: ignore[arg-type]


class TestTextsDigest:
    def test_is_deterministic(self) -> None:
        assert texts_digest(["a", "b"]) == texts_digest(["a", "b"])

    def test_order_matters(self) -> None:
        assert texts_digest(["a", "b"]) != texts_digest(["b", "a"])

    def test_boundaries_are_unambiguous(self) -> None:
        """Sin separador, ["ab"] y ["a","b"] compartirían huella."""
        assert texts_digest(["ab"]) != texts_digest(["a", "b"])

    def test_changing_the_text_changes_the_key(self) -> None:
        """Cambiar de estrategia de composición debe invalidar la caché."""
        raw = texts_digest(["Taladro. Marca: Einhell. Características: …"])
        composed = texts_digest(["Taladro. Marca: Einhell"])
        assert raw != composed


class TestEmbeddingMatrix:
    def build(self, count: int = 3, dimension: int = 4) -> EmbeddingMatrix:
        vectors = np.eye(count, dimension, dtype=np.float32)
        return EmbeddingMatrix(
            vectors=vectors,
            model_id="intfloat/multilingual-e5-small",
            role="document",
            normalized=True,
            generated_at="2026-08-01T00:00:00+00:00",
            text_sha256="abc123",
        )

    def test_metadata_records_everything_needed_to_reproduce(self) -> None:
        metadata = self.build().metadata()
        assert metadata["model_id"] == "intfloat/multilingual-e5-small"
        assert metadata["dimension"] == 4
        assert metadata["count"] == 3
        assert metadata["normalized"] is True
        assert metadata["query_prefix"] == QUERY_PREFIX
        assert metadata["text_sha256"] == "abc123"

    def test_non_e5_models_declare_no_prefix(self) -> None:
        matrix = EmbeddingMatrix(
            vectors=np.eye(2, 3, dtype=np.float32),
            model_id="BAAI/bge-small-en",
            role="document",
            normalized=True,
            generated_at="",
            text_sha256="",
        )
        assert matrix.metadata()["document_prefix"] is None

    def test_float64_is_rejected(self) -> None:
        with pytest.raises(EmbeddingError, match="float32"):
            EmbeddingMatrix(
                vectors=np.eye(2, dtype=np.float64),
                model_id="m",
                role="query",
                normalized=True,
                generated_at="",
                text_sha256="",
            )


class TestEncoderCaching:
    def test_cache_path_depends_on_model_role_and_text(self, tmp_path) -> None:
        encoder = Encoder("intfloat/multilingual-e5-small", cache_directory=tmp_path)
        other = Encoder("intfloat/multilingual-e5-base", cache_directory=tmp_path)
        texts = ["taladro", "sierra"]

        assert encoder.cache_path(texts, role="query") != encoder.cache_path(
            texts, role="document"
        )
        assert encoder.cache_path(texts, role="query") != other.cache_path(
            texts, role="query"
        )
        assert encoder.cache_path(texts, role="query") != encoder.cache_path(
            ["taladro", "lijadora"], role="query"
        )

    def test_declared_dimension_comes_from_config(self) -> None:
        assert Encoder("intfloat/multilingual-e5-small").expected_dimension == 384
        assert Encoder("intfloat/multilingual-e5-base").expected_dimension == 768
        assert Encoder("acme/desconocido").expected_dimension is None

    def test_a_corrupt_cache_is_ignored_not_fatal(self, tmp_path) -> None:
        """La caché es rendimiento: si está rota se regenera, no rompe la ejecución."""
        encoder = Encoder("intfloat/multilingual-e5-small", cache_directory=tmp_path)
        path = encoder.cache_path(["taladro"], role="query")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"esto no es un npz")
        assert encoder._read_cache(path) is None

    def test_an_invalid_batch_size_is_rejected(self) -> None:
        with pytest.raises(EmbeddingError, match="batch_size"):
            Encoder("intfloat/multilingual-e5-small", batch_size=0)


@pytest.mark.slow
class TestRealModel:
    """Requires downloading the model on first run."""

    def test_encodes_normalized_float32_vectors(self, tmp_path) -> None:
        encoder = Encoder(
            "intfloat/multilingual-e5-small", cache_directory=tmp_path, batch_size=8
        )
        matrix = encoder.encode(
            ["taladro inalámbrico", "vestido de fiesta"], role="document"
        )
        assert matrix.vectors.shape == (2, 384)
        assert matrix.vectors.dtype == np.float32
        assert np.allclose(np.linalg.norm(matrix.vectors, axis=1), 1.0, atol=1e-3)

    def test_semantic_similarity_beats_lexical_overlap(self, tmp_path) -> None:
        """La razón de ser del proyecto, en dos frases sin palabras en común."""
        encoder = Encoder("intfloat/multilingual-e5-small", cache_directory=tmp_path)
        documents = encoder.encode(
            [
                "Taladro inalámbrico 24V con dos baterías",
                "Vestido largo de fiesta para mujer",
            ],
            role="document",
        )
        query = encoder.encode(
            ["herramienta sin cable para hacer agujeros"], role="query"
        )
        similarities = documents.vectors @ query.vectors[0]
        assert similarities[0] > similarities[1]

    def test_the_cache_round_trips(self, tmp_path) -> None:
        encoder = Encoder("intfloat/multilingual-e5-small", cache_directory=tmp_path)
        texts = ["primera ficha", "segunda ficha"]
        first = encoder.encode(texts, role="document")
        assert encoder.cache_path(texts, role="document").is_file()
        second = encoder.encode(texts, role="document")
        np.testing.assert_array_equal(first.vectors, second.vectors)
        assert second.text_sha256 == first.text_sha256
