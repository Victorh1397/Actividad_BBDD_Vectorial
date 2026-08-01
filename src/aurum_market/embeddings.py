"""Encoding text into vectors, honouring the model's own contract (RF-04).

Two things are easy to get wrong and expensive to detect:

* **The E5 prefixes.** Models of the E5 family are trained with ``query: `` on
  questions and ``passage: `` on documents. They are part of the model's input
  contract, not language, so they are applied automatically and only to E5
  models. Forgetting them degrades retrieval quietly — nothing errors, results
  are just worse.
* **Normalisation.** With L2-normalised vectors the inner product equals the
  cosine, which is what makes the exact NumPy oracle a valid ground truth for a
  cosine-configured index (see 02_plan.md §3).

Encoding 15.000 products takes minutes, so results are cached on disk keyed by
everything that could change them.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .config import ARTIFACTS_DIRECTORY, EMBEDDING_DIMENSIONS

TextRole = Literal["query", "document"]

# Coincide con multilingual-e5-small, -base, -large y variantes como e5-mistral.
E5_PATTERN = re.compile(r"(?:^|[/_-])e5(?:[-_.]|$)", flags=re.IGNORECASE)

QUERY_PREFIX = "query: "
DOCUMENT_PREFIX = "passage: "


class EmbeddingError(RuntimeError):
    """Raised when text cannot be encoded into usable vectors."""


def is_e5_model(model_id: str) -> bool:
    """Return whether ``model_id`` follows the E5 input-prefix contract."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise EmbeddingError("El identificador del modelo no puede estar vacío")
    return E5_PATTERN.search(model_id.strip()) is not None


def apply_prefix(
    texts: Sequence[str], *, model_id: str, role: TextRole
) -> tuple[str, ...]:
    """Prepend the role prefix required by the E5 family, and only by it.

    TF-IDF and non-E5 models deliberately receive the original text: a prefix
    is a contract of one model family, not content.
    """
    if role not in ("query", "document"):
        raise EmbeddingError(f"El rol debe ser 'query' o 'document', no {role!r}")
    if isinstance(texts, str):
        raise EmbeddingError("Se esperaba una secuencia de textos, no una cadena")
    values = tuple(texts)
    if not values:
        raise EmbeddingError("No hay textos que codificar")
    if any(not isinstance(text, str) or not text.strip() for text in values):
        raise EmbeddingError("Todos los textos deben ser cadenas no vacías")
    if not is_e5_model(model_id):
        return values

    prefix = QUERY_PREFIX if role == "query" else DOCUMENT_PREFIX
    prefixed = []
    for text in values:
        stripped = text.strip()
        # Idempotente: aplicarlo dos veces no duplica el prefijo.
        if stripped.lower().startswith(prefix):
            prefixed.append(stripped)
        else:
            prefixed.append(f"{prefix}{stripped}")
    return tuple(prefixed)


@dataclass(frozen=True, slots=True)
class EmbeddingMatrix:
    """Vectors plus the provenance needed to trust and reproduce them."""

    vectors: NDArray[np.float32]
    model_id: str
    role: TextRole
    normalized: bool
    generated_at: str
    text_sha256: str

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise EmbeddingError(f"Se esperaba una matriz 2D, no {self.vectors.shape}")
        if self.vectors.dtype != np.float32:
            raise EmbeddingError(
                f"Los vectores deben ser float32, no {self.vectors.dtype}"
            )

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    def metadata(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "role": self.role,
            "normalized": self.normalized,
            "dimension": self.dimension,
            "count": int(self.vectors.shape[0]),
            "dtype": str(self.vectors.dtype),
            "query_prefix": QUERY_PREFIX if is_e5_model(self.model_id) else None,
            "document_prefix": DOCUMENT_PREFIX if is_e5_model(self.model_id) else None,
            "text_sha256": self.text_sha256,
            "generated_at": self.generated_at,
        }


def texts_digest(texts: Sequence[str]) -> str:
    """Return a stable digest of the exact texts encoded.

    The cache key must cover the text itself: changing the composition strategy
    changes the input while the model stays the same, and reusing stale vectors
    would silently invalidate an experiment.
    """
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class Encoder:
    """Sentence-transformers encoder with disk caching.

    The model is loaded lazily, so building an ``Encoder`` is free and a cache
    hit never pays the cost of loading weights.
    """

    def __init__(
        self,
        model_id: str = "intfloat/multilingual-e5-small",
        *,
        batch_size: int = 64,
        cache_directory: Path | None = None,
    ) -> None:
        if not model_id.strip():
            raise EmbeddingError("AURUM_EMBEDDING_MODEL no puede estar vacío")
        if batch_size < 1:
            raise EmbeddingError("batch_size debe ser positivo")
        self.model_id = model_id.strip()
        self.batch_size = batch_size
        self._cache_directory = cache_directory or ARTIFACTS_DIRECTORY / "embeddings"
        self._model = None

    @property
    def expected_dimension(self) -> int | None:
        """Declared dimension, when the model is a known one."""
        return EMBEDDING_DIMENSIONS.get(self.model_id)

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:  # pragma: no cover - dependencia declarada
                raise EmbeddingError(
                    "Falta sentence-transformers. Ejecuta `make setup`."
                ) from error
            try:
                self._model = SentenceTransformer(self.model_id)
            except Exception as error:
                raise EmbeddingError(
                    f"No se pudo cargar el modelo {self.model_id!r}: {error}. "
                    "La primera ejecución necesita conexión para descargarlo."
                ) from error
        return self._model

    def cache_path(self, texts: Sequence[str], *, role: TextRole) -> Path:
        """Return the cache location for these exact texts under this model."""
        slug = re.sub(r"[^a-z0-9]+", "-", self.model_id.lower()).strip("-")
        return self._cache_directory / f"{slug}__{role}__{texts_digest(texts)[:16]}.npz"

    def encode(
        self,
        texts: Sequence[str],
        *,
        role: TextRole,
        use_cache: bool = True,
        show_progress: bool = False,
    ) -> EmbeddingMatrix:
        """Encode ``texts`` into L2-normalised float32 vectors."""
        values = tuple(texts)
        if not values:
            raise EmbeddingError("No hay textos que codificar")

        path = self.cache_path(values, role=role)
        if use_cache and path.is_file():
            cached = self._read_cache(path)
            if cached is not None:
                return cached

        prefixed = apply_prefix(values, model_id=self.model_id, role=role)
        model = self._load_model()
        vectors = model.encode(
            list(prefixed),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise EmbeddingError(
                "El modelo no devolvió vectores normalizados pese a pedirlo"
            )
        declared = self.expected_dimension
        if declared is not None and vectors.shape[1] != declared:
            raise EmbeddingError(
                f"El modelo {self.model_id!r} devolvió dimensión {vectors.shape[1]} "
                f"pero config.py declara {declared}"
            )

        matrix = EmbeddingMatrix(
            vectors=vectors,
            model_id=self.model_id,
            role=role,
            normalized=True,
            generated_at=datetime.now(UTC).isoformat(),
            text_sha256=texts_digest(values),
        )
        if use_cache:
            self._write_cache(path, matrix)
        return matrix

    def _read_cache(self, path: Path) -> EmbeddingMatrix | None:
        try:
            with np.load(path, allow_pickle=False) as payload:
                vectors = payload["vectors"].astype(np.float32)
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError):
            # Una caché corrupta se ignora y se regenera: nunca hace fallar la
            # ejecución, porque es un detalle de rendimiento.
            return None
        if metadata.get("model_id") != self.model_id:
            return None
        return EmbeddingMatrix(
            vectors=vectors,
            model_id=metadata["model_id"],
            role=metadata["role"],
            normalized=bool(metadata.get("normalized", True)),
            generated_at=metadata.get("generated_at", ""),
            text_sha256=metadata.get("text_sha256", ""),
        )

    def _write_cache(self, path: Path, matrix: EmbeddingMatrix) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=matrix.vectors)
        path.with_suffix(".json").write_text(
            json.dumps(matrix.metadata(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
