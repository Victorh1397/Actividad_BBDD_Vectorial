"""Validated settings loaded from the environment.

Configuration fails at start-up, never halfway through an ingest. Every value
that shapes the final run is readable from one place so the execution stays
auditable (RF-18, principio P-11).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
DATA_DIRECTORY: Final = PROJECT_ROOT / "datos"
ARTIFACTS_DIRECTORY: Final = PROJECT_ROOT / ".artifacts"
RESULTS_DIRECTORY: Final = PROJECT_ROOT / "resultados"
CONFIG_DIRECTORY: Final = PROJECT_ROOT / "config"
CONTRACTS_DIRECTORY: Final = PROJECT_ROOT / "specs" / "contracts"

# Todo recurso administrable por esta actividad empieza por este prefijo. Una
# operación destructiva sobre cualquier otro nombre se rechaza (P-11).
RESOURCE_PREFIX: Final = "aurum-market"

# Contrato de IDs declarado en datos/manifest.json. Se valida, no se recalcula:
# el record_id ya viene en el catálogo.
RECORD_NAMESPACE: Final = "34ef9344-7a3f-5eb2-b30b-aceff745758d"

# Mapeo ESCI fijado por el enunciado. No es configurable.
RELEVANCE_MAPPING: Final = {"E": 3.0, "S": 2.0, "C": 1.0, "I": 0.0}

# Frontera entre relevante y no relevante para Recall@10 y MRR@10. Ver
# specs/decisiones/ADR-004-umbral-de-relevancia.md (P-05).
RELEVANCE_THRESHOLD: Final = 2.0

EMBEDDING_DIMENSIONS: Final = {
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
}


class ConfigurationError(RuntimeError):
    """Raised when the environment cannot support a correct run."""


@dataclass(frozen=True, slots=True)
class HnswSettings:
    """Graph parameters declared explicitly so the final run is auditable (RF-08).

    ``indexing_threshold`` and ``full_scan_threshold`` are not cosmetic. Qdrant
    defaults both to 10.000 because below that size a linear scan beats walking
    a graph, so a collection of 15.000 products split across segments may never
    build an HNSW index at all. That would leave ``m`` and ``ef_construct``
    without observable effect and make ANN fidelity trivially 1.0 — measuring
    an index that is not being used. Lowering them forces the index to exist.
    """

    m: int = 24
    ef_construct: int = 120
    ef_search: int = 128
    indexing_threshold: int = 1_000
    full_scan_threshold: int = 1_000

    def __post_init__(self) -> None:
        if self.m < 2:
            raise ConfigurationError("AURUM_HNSW_M debe ser >= 2")
        if self.ef_construct < 4:
            raise ConfigurationError("AURUM_HNSW_EF_CONSTRUCT debe ser >= 4")
        if self.ef_search < 1:
            raise ConfigurationError("AURUM_HNSW_EF_SEARCH debe ser >= 1")
        if self.indexing_threshold < 0:
            raise ConfigurationError("AURUM_INDEXING_THRESHOLD no puede ser negativo")
        if self.full_scan_threshold < 0:
            raise ConfigurationError("AURUM_FULL_SCAN_THRESHOLD no puede ser negativo")

    def as_dict(self) -> dict[str, int]:
        return {
            "m": self.m,
            "ef_construct": self.ef_construct,
            "ef_search": self.ef_search,
            "indexing_threshold": self.indexing_threshold,
            "full_scan_threshold": self.full_scan_threshold,
        }


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the pipeline needs to run, validated on construction."""

    embedding_model: str
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    batch_size: int
    allow_reset: bool
    confirm_cleanup: str
    hnsw: HnswSettings = field(default_factory=HnswSettings)

    def __post_init__(self) -> None:
        if not self.embedding_model.strip():
            raise ConfigurationError("AURUM_EMBEDDING_MODEL no puede estar vacío")
        if not self.qdrant_url.startswith(("http://", "https://")):
            raise ConfigurationError(
                f"QDRANT_URL debe empezar por http:// o https://, recibido {self.qdrant_url!r}"
            )
        if self.batch_size < 1:
            raise ConfigurationError("AURUM_BATCH_SIZE debe ser un entero positivo")
        validate_resource_name(self.qdrant_collection)

    @property
    def embedding_dimension(self) -> int:
        """Return the declared dimension for the configured model."""
        try:
            return EMBEDDING_DIMENSIONS[self.embedding_model]
        except KeyError as error:
            known = ", ".join(sorted(EMBEDDING_DIMENSIONS))
            raise ConfigurationError(
                f"Dimensión desconocida para {self.embedding_model!r}. "
                f"Modelos declarados: {known}. Añádelo a EMBEDDING_DIMENSIONS."
            ) from error

    def cleanup_authorized(self, resource_name: str) -> bool:
        """Return whether a destructive operation on ``resource_name`` is allowed.

        Requires both the permission flag and the exact resource name typed by
        hand. Neither alone is enough (P-11).
        """
        validate_resource_name(resource_name)
        return self.allow_reset and self.confirm_cleanup.strip() == resource_name


def validate_resource_name(resource_name: str) -> str:
    """Reject administrative operations outside this activity's namespace.

    Adapted from ``operations.validate_resource_name`` of session 03.
    """
    normalized = re.sub(r"[^a-z0-9]", "", resource_name.lower())
    expected = re.sub(r"[^a-z0-9]", "", RESOURCE_PREFIX)
    if not normalized.startswith(expected):
        raise ConfigurationError(
            f"El recurso {resource_name!r} queda fuera del prefijo protegido "
            f"{RESOURCE_PREFIX!r}. Ninguna operación puede tocarlo."
        )
    return resource_name


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ConfigurationError(
            f"{name} debe ser un entero, recibido {raw!r}"
        ) from error


def _read_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"true", "1", "yes", "y", "on"}:
        return True
    if raw in {"false", "0", "no", "n", "off"}:
        return False
    raise ConfigurationError(f"{name} debe ser true o false, recibido {raw!r}")


def load_settings(*, env_file: Path | None = None) -> Settings:
    """Load and validate settings from ``.env`` plus the process environment."""
    load_dotenv(env_file or PROJECT_ROOT / ".env")
    api_key = os.getenv("QDRANT_API_KEY", "").strip()
    return Settings(
        embedding_model=os.getenv(
            "AURUM_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
        ).strip(),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333").strip(),
        qdrant_api_key=api_key or None,
        qdrant_collection=os.getenv(
            "QDRANT_COLLECTION", f"{RESOURCE_PREFIX}-catalogo"
        ).strip(),
        batch_size=_read_int("AURUM_BATCH_SIZE", 256),
        allow_reset=_read_bool("AURUM_ALLOW_RESET", default=False),
        confirm_cleanup=os.getenv("AURUM_CONFIRM_CLEANUP", ""),
        hnsw=HnswSettings(
            m=_read_int("AURUM_HNSW_M", 24),
            ef_construct=_read_int("AURUM_HNSW_EF_CONSTRUCT", 120),
            ef_search=_read_int("AURUM_HNSW_EF_SEARCH", 128),
            indexing_threshold=_read_int("AURUM_INDEXING_THRESHOLD", 1_000),
            full_scan_threshold=_read_int("AURUM_FULL_SCAN_THRESHOLD", 1_000),
        ),
    )
