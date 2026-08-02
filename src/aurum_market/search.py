"""The common retrieval interface (RF-01, RF-13, RF-14).

Every retriever in the system answers the same question the same way: given a
query in natural language, return a ranked list of ``SearchHit``. TF-IDF does it
by matching words, the dense retriever by comparing meaning, and later Qdrant
will do it through an ANN index — but the evaluation code cannot tell them
apart, which is precisely what allows a failure to be attributed to one layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .config import Settings
from .contracts import SearchHit
from .embeddings import Encoder
from .store.base import StoreError, VectorStore


@runtime_checkable
class Retriever(Protocol):
    """Anything that turns a query string into a ranked list of products."""

    def search(
        self, query_text: str, *, top_k: int = 10, brand: str | None = None
    ) -> list[SearchHit]:
        """Return the ``top_k`` best products, optionally within one brand."""
        ...


class DenseRetriever:
    """Encodes the query and searches a vector store.

    The store never sees text and the encoder never sees products. That
    separation is what lets the very same retriever run against the exact
    NumPy oracle or against Qdrant, which in turn is what makes ANN fidelity
    measurable: both paths differ only in the store (RF-20).
    """

    def __init__(self, store: VectorStore, encoder: Encoder) -> None:
        self._store = store
        self._encoder = encoder
        declared = encoder.expected_dimension
        if declared is not None and store.size and store.dimension != declared:
            raise StoreError(
                f"El almacén tiene dimensión {store.dimension} y el modelo "
                f"{encoder.model_id!r} produce {declared}. Los vectores del "
                "catálogo se generaron con otro modelo."
            )

    @property
    def size(self) -> int:
        return self._store.size

    @property
    def model_id(self) -> str:
        return self._encoder.model_id

    def search(
        self, query_text: str, *, top_k: int = 10, brand: str | None = None
    ) -> list[SearchHit]:
        """Encode the query with the ``query:`` role and retrieve."""
        if not isinstance(query_text, str) or not query_text.strip():
            raise StoreError("La consulta no puede estar vacía")
        # El rol importa: E5 se entrena con prefijos distintos para consultas y
        # documentos, y codificar una consulta como documento degrada el ranking
        # sin que nada falle visiblemente (RF-04).
        matrix = self._encoder.encode([query_text], role="query")
        return self._store.search_vector(matrix.vectors[0], top_k=top_k, brand=brand)

    def search_many(
        self, query_texts: Sequence[str], *, top_k: int = 10
    ) -> list[list[SearchHit]]:
        """Retrieve for several queries, encoding them in a single batch.

        Encoding one query at a time wastes the model's batching, which matters
        when timing: the cost of encoding would otherwise be mixed into what
        looks like search latency (RF-21).
        """
        texts = list(query_texts)
        if not texts:
            return []
        matrix = self._encoder.encode(texts, role="query")
        return [
            self._store.search_vector(matrix.vectors[position], top_k=top_k)
            for position in range(len(texts))
        ]


def build_live_retriever(
    settings: Settings, *, expected_points: int | None = None
) -> DenseRetriever:
    """Build the retriever the delivery actually uses: Qdrant plus the encoder.

    Verifies the collection before returning it, so a search never runs against
    a half-ingested collection and produces numbers that look plausible (RF-10).
    """
    from .ingest import verify_collection
    from .store.qdrant_store import QdrantStore

    store = QdrantStore(settings)
    status = store.status()
    if not status.exists or status.points_count == 0:
        raise StoreError(
            f"La colección {settings.qdrant_collection!r} no está lista. "
            "Ejecuta `aurum ingest`."
        )
    if expected_points is not None:
        verify_collection(
            store,
            expected_points=expected_points,
            expected_dimension=settings.embedding_dimension,
        )
    return DenseRetriever(store, Encoder(settings.embedding_model))
