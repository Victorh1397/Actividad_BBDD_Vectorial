"""Qdrant collection through its native SDK (RF-07, RF-08, RF-14, RF-18).

The engine is configured explicitly, never by default: dimension, distance,
HNSW graph parameters and a payload index on ``brand``. That last one is what
turns brand filtering into a real database operation instead of a post-filter
over global results (P-09).

Point IDs are the catalog's UUIDv5, which is what makes ingestion idempotent by
construction: re-running it overwrites each product in place rather than adding
a copy (P-08).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from time import monotonic, sleep

import numpy as np
from numpy.typing import ArrayLike

from ..config import Settings, validate_resource_name
from ..contracts import CatalogRecord, SearchHit
from .base import CollectionEmptyError, ProviderUnavailableError, StoreError


@dataclass(frozen=True, slots=True)
class CollectionStatus:
    """What the engine reports about the collection right now.

    The HNSW fields are read back from Qdrant rather than assumed: declaring a
    configuration and verifying that the engine applied it are different things,
    and RF-08 asks for the second (P-06).
    """

    exists: bool
    points_count: int
    indexed_vectors_count: int
    dimension: int | None = None
    distance: str = ""
    status: str = ""
    segments_count: int = 0
    hnsw_m: int | None = None
    hnsw_ef_construct: int | None = None
    full_scan_threshold: int | None = None
    indexing_threshold: int | None = None

    @property
    def fully_indexed(self) -> bool:
        """Whether every stored vector is searchable through the index.

        Qdrant keeps small segments unindexed on purpose, so this being false
        is not an error by itself — but it must be observed before trusting a
        latency measurement (RF-10).
        """
        return self.exists and self.indexed_vectors_count >= self.points_count

    @property
    def kilobytes_per_segment(self) -> float:
        """Approximate segment size, which is what the thresholds compare against.

        Qdrant decides whether to build the graph per segment, not per
        collection, so a collection can sit above the threshold in total and
        still be answered by brute force.
        """
        if not self.dimension or self.segments_count < 1:
            return 0.0
        return self.points_count * self.dimension * 4 / 1024 / self.segments_count


class QdrantStore:
    """Native-SDK wrapper over one Qdrant collection."""

    def __init__(self, settings: Settings, *, timeout: float = 60.0) -> None:
        self._settings = settings
        self.collection = validate_resource_name(settings.qdrant_collection)
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as error:  # pragma: no cover - dependencia declarada
            raise ProviderUnavailableError(
                "Falta qdrant-client. Ejecuta `make setup`."
            ) from error
        self._models = models
        self._client = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=timeout
        )

    # ---------------------------------------------------------------- estado

    def health(self) -> None:
        """Fail early with an actionable message if the engine is unreachable."""
        try:
            self._client.get_collections()
        except Exception as error:
            raise ProviderUnavailableError(
                f"No se puede alcanzar Qdrant en {self._settings.qdrant_url}: "
                f"{type(error).__name__}. Arranca Docker Desktop y ejecuta `make up`."
            ) from error

    def status(self) -> CollectionStatus:
        """Report counts and indexing state without raising on absence."""
        self.health()
        if not self._client.collection_exists(self.collection):
            return CollectionStatus(
                exists=False, points_count=0, indexed_vectors_count=0
            )
        info = self._client.get_collection(self.collection)
        params = info.config.params.vectors
        hnsw = info.config.hnsw_config
        optimizers = info.config.optimizer_config
        return CollectionStatus(
            exists=True,
            points_count=int(info.points_count or 0),
            indexed_vectors_count=int(info.indexed_vectors_count or 0),
            dimension=int(getattr(params, "size", 0)) or None,
            distance=str(getattr(params, "distance", "")),
            status=str(info.status),
            segments_count=int(info.segments_count or 0),
            hnsw_m=getattr(hnsw, "m", None),
            hnsw_ef_construct=getattr(hnsw, "ef_construct", None),
            full_scan_threshold=getattr(hnsw, "full_scan_threshold", None),
            indexing_threshold=getattr(optimizers, "indexing_threshold", None),
        )

    @property
    def size(self) -> int:
        return self.status().points_count

    # ------------------------------------------------------------- esquema

    def ensure_collection(self, *, dimension: int, recreate: bool = False) -> bool:
        """Create the collection if needed. Returns whether it was created.

        ``recreate`` is destructive and therefore goes through the same double
        authorisation as ``reset`` (P-11).
        """
        self.health()
        models = self._models

        if self._client.collection_exists(self.collection):
            if not recreate:
                current = self.status()
                if current.dimension not in (None, dimension):
                    raise StoreError(
                        f"La colección {self.collection!r} tiene dimensión "
                        f"{current.dimension} y el modelo produce {dimension}. "
                        "La dimensión es parte del esquema: hay que recrearla "
                        "con `aurum reset` antes de cambiar de modelo."
                    )
                return False
            self.reset()

        hnsw = self._settings.hnsw
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE
            ),
            # m y ef_construct gobiernan el grafo y no pueden cambiarse sin
            # reindexar; ef_search sí se ajusta por consulta (RF-08).
            hnsw_config=models.HnswConfigDiff(
                m=hnsw.m,
                ef_construct=hnsw.ef_construct,
                # En KILOBYTES, no en número de vectores. Por debajo de este
                # tamaño Qdrant responde por fuerza bruta aunque exista grafo.
                full_scan_threshold=hnsw.full_scan_threshold,
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=hnsw.indexing_threshold
            ),
        )
        # Sin este índice el filtro por marca sería un recorrido lineal del
        # payload. Con él, el filtro forma parte de la consulta (RF-14).
        self._client.create_payload_index(
            collection_name=self.collection,
            field_name="brand",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        return True

    def reset(self) -> None:
        """Delete the collection. Requires explicit double authorisation."""
        if not self._settings.cleanup_authorized(self.collection):
            raise StoreError(
                f"Operación destructiva bloqueada sobre {self.collection!r}. "
                "Requiere AURUM_ALLOW_RESET=true y AURUM_CONFIRM_CLEANUP con el "
                "nombre exacto de la colección."
            )
        self.health()
        if self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)

    # ------------------------------------------------------------- escritura

    def upsert_batch(self, records: Sequence[CatalogRecord], vectors: ArrayLike) -> int:
        """Write one batch, keyed by ``record_id``. Returns how many were sent."""
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(records) != matrix.shape[0]:
            raise StoreError(
                f"Desalineación: {len(records)} registros y {matrix.shape[0]} vectores"
            )
        if not records:
            return 0

        points = [
            self._models.PointStruct(
                # El UUIDv5 del catálogo es el ID del punto: repetir la ingesta
                # sobrescribe en lugar de duplicar (P-08).
                id=record.record_id,
                vector=matrix[position].tolist(),
                payload=record.payload(),
            )
            for position, record in enumerate(records)
        ]
        self._client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def delete_by_record_ids(self, record_ids: Iterable[str]) -> int:
        """Remove points by ID. Returns how many were requested."""
        ids = [str(value) for value in record_ids]
        if not ids:
            return 0
        self._client.delete(
            collection_name=self.collection,
            points_selector=self._models.PointIdsList(points=ids),
            wait=True,
        )
        return len(ids)

    # ------------------------------------------------------------- lectura

    def get_by_record_id(self, record_id: str) -> dict[str, object] | None:
        """Read one point's payload, or ``None`` when it is not there."""
        self.health()
        found = self._client.retrieve(
            collection_name=self.collection, ids=[record_id], with_payload=True
        )
        return dict(found[0].payload or {}) if found else None

    def search_vector(
        self,
        query_vector: ArrayLike,
        *,
        top_k: int = 10,
        brand: str | None = None,
        ef_search: int | None = None,
    ) -> list[SearchHit]:
        """Search the collection, optionally restricted to one brand."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise StoreError(f"top_k debe ser un entero positivo, recibido {top_k!r}")
        status = self.status()
        if not status.exists:
            raise CollectionEmptyError(
                f"La colección {self.collection!r} no existe. Ejecuta `aurum ingest`."
            )
        if status.points_count == 0:
            raise CollectionEmptyError(
                f"La colección {self.collection!r} está vacía. Ejecuta `aurum ingest`."
            )

        models = self._models
        vector = np.ascontiguousarray(query_vector, dtype=np.float32).ravel()

        query_filter = None
        if brand is not None:
            # El filtro viaja DENTRO de la consulta: Qdrant restringe el grafo
            # a los puntos que cumplen la condición (P-09).
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="brand", match=models.MatchValue(value=brand)
                    )
                ]
            )

        response = self._client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            search_params=models.SearchParams(
                hnsw_ef=ef_search or self._settings.hnsw.ef_search
            ),
        )
        return [
            self._build_hit(point, rank)
            for rank, point in enumerate(response.points, start=1)
        ]

    def _build_hit(self, point: object, rank: int) -> SearchHit:
        payload = dict(getattr(point, "payload", None) or {})
        return SearchHit(
            rank=rank,
            record_id=str(getattr(point, "id", "")),
            product_id=str(payload.get("product_id", "")),
            title=str(payload.get("title", "")),
            brand=str(payload.get("brand", "")),
            color=str(payload.get("color", "")),
            native_score=float(getattr(point, "score", 0.0)),
            # Qdrant con Distance.COSINE devuelve una SIMILITUD en [-1, 1],
            # no una distancia: más alto es mejor (P-03).
            score_kind="similarity",
            higher_is_better=True,
        )

    # ------------------------------------------------------------- espera

    def wait_until_indexed(
        self,
        *,
        expected_points: int,
        timeout_seconds: float = 120.0,
        require_indexed: bool = False,
    ) -> CollectionStatus:
        """Poll until the collection holds —and optionally indexes— what it should.

        A confirmed write must become observable, and if it does not the system
        must say so instead of carrying on (P-10). Adapted from ``wait_until``
        of session 03.

        ``require_indexed`` matters because Qdrant reports ``green`` as soon as
        the points are stored and builds the HNSW graph afterwards, in the
        background. Waiting only for ``green`` therefore proves the data is
        there but says nothing about whether the index exists — and a latency
        measured before the graph is built is measuring a brute-force scan.
        """
        if timeout_seconds <= 0:
            raise StoreError("El plazo de espera debe ser positivo")
        started = monotonic()
        while True:
            status = self.status()
            elapsed = monotonic() - started
            stored = status.points_count >= expected_points and status.status == "green"
            indexed = (
                not require_indexed or status.indexed_vectors_count >= expected_points
            )
            if stored and indexed:
                return status
            if elapsed >= timeout_seconds:
                pending = (
                    "El índice no ha terminado de construirse"
                    if stored
                    else "La escritura no se ha vuelto observable"
                )
                raise StoreError(
                    f"Tras {elapsed:.1f}s la colección tiene "
                    f"{status.points_count} puntos (esperados {expected_points}), "
                    f"{status.indexed_vectors_count} indexados y estado "
                    f"{status.status!r}. {pending}."
                )
            sleep(0.5)
