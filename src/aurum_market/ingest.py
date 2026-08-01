"""Batched, idempotent ingestion into the vector store (RF-09, RF-10).

Idempotence here is structural, not a clean-up step: every product is written
under its catalog UUIDv5, so a second run overwrites each point in place. There
is no de-duplication pass because there is nothing to de-duplicate (P-08).

The count is verified before the collection is declared usable: an ingestion
that silently dropped a batch would otherwise surface much later, as unexplained
metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from .contracts import CatalogRecord, CatalogSnapshot, Profile
from .embeddings import Encoder
from .store.base import StoreError
from .store.qdrant_store import CollectionStatus, QdrantStore
from .text import TextStrategy, compose_all


@dataclass(slots=True)
class IngestReport:
    """What one ingestion run did, in enough detail to audit it."""

    profile: Profile
    records: int
    batches: int
    sent: int
    created_collection: bool
    text_strategy: TextStrategy
    embedding_model: str
    dimension: int
    status: CollectionStatus | None = None
    notes: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def points_count(self) -> int:
        return self.status.points_count if self.status else 0

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "records": self.records,
            "batches": self.batches,
            "sent": self.sent,
            "created_collection": self.created_collection,
            "text_strategy": self.text_strategy,
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "points_count": self.points_count,
            "indexed_vectors_count": (
                self.status.indexed_vectors_count if self.status else 0
            ),
            "collection_status": self.status.status if self.status else "",
            "notes": self.notes,
            "generated_at": self.generated_at,
        }


def iter_batches(
    records: Sequence[CatalogRecord], *, batch_size: int
) -> Iterator[tuple[int, Sequence[CatalogRecord]]]:
    """Yield ``(offset, batch)`` pairs, deterministically."""
    if batch_size < 1:
        raise StoreError(f"batch_size debe ser positivo, recibido {batch_size}")
    for start in range(0, len(records), batch_size):
        yield start, records[start : start + batch_size]


def ingest_catalog(
    catalog: CatalogSnapshot,
    store: QdrantStore,
    encoder: Encoder,
    *,
    text_strategy: TextStrategy = "title_brand_color",
    batch_size: int = 256,
    show_progress: bool = False,
    on_batch: Callable[[int, int], None] | None = None,
) -> IngestReport:
    """Encode and upsert an entire catalog profile, then verify the result."""
    if not len(catalog):
        raise StoreError("El catálogo está vacío: no hay nada que ingerir")

    texts = compose_all(catalog.records, text_strategy)
    matrix = encoder.encode(texts, role="document", show_progress=show_progress)
    dimension = matrix.dimension

    created = store.ensure_collection(dimension=dimension)

    sent = 0
    batches = 0
    for offset, batch in iter_batches(catalog.records, batch_size=batch_size):
        vectors = matrix.vectors[offset : offset + len(batch)]
        sent += store.upsert_batch(batch, vectors)
        batches += 1
        if on_batch is not None:
            on_batch(sent, len(catalog))

    if sent != len(catalog):
        raise StoreError(
            f"Se enviaron {sent} puntos de {len(catalog)}: la ingesta perdió datos"
        )

    # Una escritura confirmada tiene que volverse observable, o el sistema lo
    # dice en lugar de seguir (P-10).
    status = store.wait_until_indexed(expected_points=len(catalog))
    if status.points_count != len(catalog):
        raise StoreError(
            f"La colección tiene {status.points_count} puntos y el catálogo "
            f"{len(catalog)}. Si hay más, algún ID dejó de ser estable; si hay "
            "menos, se perdieron escrituras."
        )

    return IngestReport(
        profile=catalog.profile,
        records=len(catalog),
        batches=batches,
        sent=sent,
        created_collection=created,
        text_strategy=text_strategy,
        embedding_model=encoder.model_id,
        dimension=dimension,
        status=status,
        notes=[
            "Los puntos se escriben bajo el record_id (UUIDv5) del catálogo, así "
            "que repetir la ingesta sobrescribe en vez de duplicar.",
        ],
    )


def verify_collection(
    store: QdrantStore, *, expected_points: int, expected_dimension: int | None = None
) -> CollectionStatus:
    """Check the collection is fit to answer queries (RF-10).

    Run before trusting any measurement: a collection that is missing points,
    has the wrong dimension, or is still indexing produces numbers that mean
    something different from what they appear to mean.
    """
    status = store.status()
    if not status.exists:
        raise StoreError(
            f"La colección {store.collection!r} no existe. Ejecuta `aurum ingest`."
        )
    if status.points_count != expected_points:
        raise StoreError(
            f"La colección tiene {status.points_count} puntos, se esperaban "
            f"{expected_points}."
        )
    if expected_dimension is not None and status.dimension != expected_dimension:
        raise StoreError(
            f"La colección tiene dimensión {status.dimension} y se esperaba "
            f"{expected_dimension}. Los vectores se generaron con otro modelo."
        )
    if status.distance.lower() != "cosine":
        raise StoreError(
            f"La colección usa distancia {status.distance!r} y el sistema asume "
            "coseno. El significado del score no sería el declarado."
        )
    return status


def embed_queries(
    encoder: Encoder, texts: Sequence[str]
) -> np.ndarray:
    """Encode query texts with the ``query:`` role."""
    return encoder.encode(list(texts), role="query").vectors
