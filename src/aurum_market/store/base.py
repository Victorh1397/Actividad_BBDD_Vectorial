"""The contract every retrieval backend honours (RF-01, RF-15).

One interface for the exact NumPy oracle and for Qdrant is what makes RF-20
possible: the same evaluation code can run against both, so any difference in
the results is attributable to the index and not to the plumbing around it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..contracts import SearchHit


class StoreError(RuntimeError):
    """Base class for every failure a retrieval backend can report."""


class CollectionEmptyError(StoreError):
    """The collection holds no vectors, so a search cannot mean anything.

    Deliberately an error and not an empty list: a silent empty result looks
    identical to "nothing matched" and would let an un-ingested collection pass
    unnoticed all the way to the metrics (RF-15).
    """


class ProviderUnavailableError(StoreError):
    """The backing engine cannot be reached. The message must say what to do."""


@runtime_checkable
class VectorStore(Protocol):
    """Minimum surface shared by every backend."""

    @property
    def size(self) -> int:
        """Number of indexed vectors."""
        ...

    def search_vector(
        self,
        query_vector: object,
        *,
        top_k: int = 10,
        brand: str | None = None,
    ) -> list[SearchHit]:
        """Return the ``top_k`` nearest records, optionally within one brand.

        Filtering by brand is part of the query, never a post-processing step
        applied to global results (P-09).
        """
        ...
