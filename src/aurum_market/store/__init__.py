"""Storage and retrieval backends behind one common interface."""

from __future__ import annotations

from .base import (
    CollectionEmptyError,
    ProviderUnavailableError,
    StoreError,
    VectorStore,
)
from .exact_store import ExactVectorStore

__all__ = [
    "CollectionEmptyError",
    "ExactVectorStore",
    "ProviderUnavailableError",
    "StoreError",
    "VectorStore",
]
