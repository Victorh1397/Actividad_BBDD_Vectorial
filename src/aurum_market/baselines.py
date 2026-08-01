"""Lexical baseline: TF-IDF over the same text the dense system encodes (RF-02).

The statement asks to start from an interpretable reference. TF-IDF matches
words, not meaning, so it marks the line the dense system has to beat: if
embeddings do not improve on plain word matching, the extra machinery is not
earning its keep.

It also makes the failure mode visible. A query like "quiero una herramienta
inalámbrica potente para perforar sin depender de un enchufe" shares almost no
vocabulary with the title of a drill, which is exactly the business problem
Aurum Market has today.

Adapted from ``TfidfRetriever`` of session 01.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .contracts import CatalogRecord, SearchHit
from .store.base import CollectionEmptyError, StoreError
from .text import TextStrategy, compose_all


class TfidfRetriever:
    """Sparse lexical retriever with L2-normalised TF-IDF vectors."""

    def __init__(
        self,
        records: Sequence[CatalogRecord],
        *,
        strategy: TextStrategy = "raw_text",
        ngram_range: tuple[int, int] = (1, 2),
        min_document_frequency: int = 1,
        max_features: int | None = 200_000,
    ) -> None:
        if not records:
            raise CollectionEmptyError("TF-IDF necesita al menos un producto")
        self._records = tuple(records)
        self._strategy = strategy
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_document_frequency,
            max_features=max_features,
            lowercase=True,
            # El catálogo está en español: sin plegar acentos, "cámara" y
            # "camara" serían términos distintos y la consulta real usa ambos.
            strip_accents="unicode",
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        texts = compose_all(self._records, strategy)
        try:
            self._matrix = self._vectorizer.fit_transform(texts)
        except ValueError as error:  # vocabulario vacío
            raise StoreError(
                "No se pudo construir el vocabulario TF-IDF. Revisa el corpus."
            ) from error
        self._brands = np.array(
            [record.brand for record in self._records], dtype=object
        )

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def vocabulary_size(self) -> int:
        return len(self._vectorizer.vocabulary_)

    @property
    def strategy(self) -> TextStrategy:
        return self._strategy

    def search(
        self,
        query_text: str,
        *,
        top_k: int = 10,
        brand: str | None = None,
    ) -> list[SearchHit]:
        """Rank products by sparse cosine similarity against the query terms."""
        if not isinstance(query_text, str) or not query_text.strip():
            raise StoreError("La consulta no puede estar vacía")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise StoreError(f"top_k debe ser un entero positivo, recibido {top_k!r}")

        query_vector = self._vectorizer.transform([query_text])
        scores = np.asarray((query_vector @ self._matrix.T).todense()).ravel()

        candidates = np.arange(self.size, dtype=np.int64)
        if brand is not None:
            candidates = candidates[self._brands == brand]
            if candidates.size == 0:
                return []
        selected = scores[candidates]

        count = min(top_k, candidates.size)
        partition = np.argpartition(-selected, count - 1)[:count]
        ordered = partition[np.lexsort((partition, -selected[partition]))]

        hits = []
        for rank, local in enumerate(ordered, start=1):
            record = self._records[int(candidates[local])]
            hits.append(
                SearchHit(
                    rank=rank,
                    record_id=record.record_id,
                    product_id=record.product_id,
                    title=record.title,
                    brand=record.brand,
                    color=record.color,
                    native_score=float(selected[local]),
                    # TF-IDF con norma L2 devuelve un coseno en [0, 1].
                    score_kind="similarity",
                    higher_is_better=True,
                )
            )
        return hits
