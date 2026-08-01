"""The common retrieval interface (RF-01, RF-13, RF-14).

Every retriever in the system answers the same question the same way: given a
query in natural language, return a ranked list of ``SearchHit``. TF-IDF does it
by matching words, the dense retriever by comparing meaning, and later Qdrant
will do it through an ANN index — but the evaluation code cannot tell them
apart, which is precisely what allows a failure to be attributed to one layer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .contracts import SearchHit
from .embeddings import Encoder
from .store.base import StoreError
from .store.exact_store import ExactVectorStore


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

    The store never sees text and the encoder never sees products: keeping the
    two apart is what lets the same store be queried by an exact oracle or by
    an ANN index without touching this class.
    """

    def __init__(self, store: ExactVectorStore, encoder: Encoder) -> None:
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
