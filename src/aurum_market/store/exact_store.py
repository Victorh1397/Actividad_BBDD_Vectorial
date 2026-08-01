"""Exact nearest-neighbour search in NumPy: the ground truth (RF-20).

With L2-normalised vectors, exact cosine search is a matrix-vector product
followed by a partial sort. On 15.000 x 384 float32 —23 MB— that is a handful
of milliseconds, so FAISS would add a heavy binary dependency without closing a
single extra requirement. See ADR-003.

Being five lines of linear algebra is the point: nobody can accuse the ground
truth of approximating anything.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..contracts import CatalogRecord, SearchHit
from .base import CollectionEmptyError, StoreError


class ExactVectorStore:
    """Brute-force cosine search over an in-memory embedding matrix."""

    def __init__(
        self,
        records: Sequence[CatalogRecord],
        embeddings: ArrayLike,
        *,
        assume_normalized: bool = True,
    ) -> None:
        matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise StoreError(
                f"Los embeddings deben ser una matriz 2D, no {matrix.shape}"
            )
        if len(records) != matrix.shape[0]:
            raise StoreError(
                f"Desalineación: {len(records)} registros frente a "
                f"{matrix.shape[0]} vectores. Los IDs dejarían de corresponderse."
            )
        if matrix.size and not np.isfinite(matrix).all():
            raise StoreError("Los embeddings contienen NaN o infinito")
        if assume_normalized and matrix.size:
            norms = np.linalg.norm(matrix, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise StoreError(
                    "Los embeddings no están L2-normalizados, así que el producto "
                    "interno no equivale al coseno. Normalízalos o pasa "
                    "assume_normalized=False."
                )
        self._records = tuple(records)
        self._embeddings = matrix
        # Marcas precalculadas: el filtro es parte de la consulta, no un
        # descarte posterior sobre resultados globales (P-09).
        self._brands = np.array(
            [record.brand for record in self._records], dtype=object
        )

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def dimension(self) -> int:
        return int(self._embeddings.shape[1]) if self._embeddings.size else 0

    def search_vector(
        self,
        query_vector: ArrayLike,
        *,
        top_k: int = 10,
        brand: str | None = None,
    ) -> list[SearchHit]:
        """Return the exact ``top_k`` nearest records by cosine similarity."""
        if self.size == 0:
            raise CollectionEmptyError(
                "El almacén exacto está vacío: no hay nada que recuperar"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise StoreError(f"top_k debe ser un entero positivo, recibido {top_k!r}")

        vector = np.ascontiguousarray(query_vector, dtype=np.float32).ravel()
        if vector.shape != (self.dimension,):
            raise StoreError(
                f"La consulta tiene dimensión {vector.shape[0]} y el índice "
                f"{self.dimension}. Probablemente son modelos distintos."
            )

        candidates = np.arange(self.size, dtype=np.int64)
        if brand is not None:
            candidates = candidates[self._brands == brand]
            if candidates.size == 0:
                # Un filtro sin resultados es una respuesta legítima, no un
                # error: la marca simplemente no está en el catálogo (RF-15).
                return []

        scores = np.asarray(self._embeddings[candidates] @ vector, dtype=np.float64)
        ordered = _stable_top_k(scores, k=min(top_k, candidates.size))

        return [
            self._build_hit(
                position=int(candidates[local]),
                score=float(scores[local]),
                rank=rank,
            )
            for rank, local in enumerate(ordered, start=1)
        ]

    def _build_hit(self, *, position: int, score: float, rank: int) -> SearchHit:
        record = self._records[position]
        return SearchHit(
            rank=rank,
            record_id=record.record_id,
            product_id=record.product_id,
            title=record.title,
            brand=record.brand,
            color=record.color,
            native_score=score,
            # Con vectores normalizados el producto interno ES el coseno, así
            # que el score es una similitud en [-1, 1] (P-03).
            score_kind="similarity",
            higher_is_better=True,
        )


def _stable_top_k(scores: NDArray[np.float64], *, k: int) -> NDArray[np.intp]:
    """Return the indices of the ``k`` highest scores, ties broken by position.

    ``argpartition`` finds the cut-off in linear time; only the selected
    candidates are then sorted. Deterministic tie-breaking matters because two
    products with identical text would otherwise swap places between runs and
    make the rankings irreproducible.
    """
    if k >= scores.size:
        return np.lexsort((np.arange(scores.size), -scores))
    partition = np.argpartition(-scores, k - 1)[:k]
    return partition[np.lexsort((partition, -scores[partition]))]
