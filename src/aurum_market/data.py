"""Single entry point for loading, validating, and sanitizing the dataset.

Two rules live here and nowhere else:

* A missing value becomes an empty string. Never ``"nan"``, ``"None"`` or
  ``"null"`` (RF-03, principio P-07).
* Counts and checksums are checked against ``datos/manifest.json`` before any
  record reaches the vector store (RF-07).
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_DIRECTORY, RECORD_NAMESPACE
from .contracts import CatalogRecord, CatalogSnapshot, Profile

# Los ficheros bajo datos/profesorado/ del manifiesto son las relevancias y
# duplicados reservados a la corrección: no forman parte de la entrega y su
# ausencia es lo correcto.
RESERVED_PREFIX = "datos/profesorado/"

# Cada perfil declara su fichero y el recuento que el manifiesto promete. La
# muestra sirve para desarrollar; el catálogo completo es el recorrido evaluado.
CATALOG_PROFILES: dict[str, tuple[str, str]] = {
    "sample": ("catalogo_muestra.csv", "sample_records"),
    "full": ("catalogo_productos.csv.gz", "catalog_records"),
}

CATALOG_COLUMNS = (
    "record_id",
    "product_id",
    "title",
    "brand",
    "color",
    "locale",
    "text",
    "catalog_version",
    "active",
)

_NAMESPACE = uuid.UUID(RECORD_NAMESPACE)

EXPECTED_FILES = (
    "catalogo_productos.csv.gz",
    "catalogo_muestra.csv",
    "consultas_desarrollo.csv",
    "consultas_evaluacion.csv",
    "consultas_filtradas.csv",
    "relevancias_desarrollo.csv",
    "eventos_catalogo.csv",
    "altas_desarrollo.csv",
    "altas_evaluacion.csv",
)


class DataIntegrityError(RuntimeError):
    """Raised when the dataset on disk does not match its manifest."""


@dataclass(frozen=True, slots=True)
class FileCheck:
    """Outcome of verifying one declared file."""

    name: str
    present: bool
    checksum_matches: bool | None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.present and self.checksum_matches is not False


def load_manifest(*, data_directory: Path | None = None) -> dict[str, object]:
    """Read the provenance manifest that governs counts, IDs and checksums."""
    directory = data_directory or DATA_DIRECTORY
    path = directory / "manifest.json"
    if not path.is_file():
        raise DataIntegrityError(
            f"No se encuentra {path}. Copia datos/ desde el material de la actividad."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data_integrity(
    *, data_directory: Path | None = None, verify_checksums: bool = True
) -> list[FileCheck]:
    """Check every expected file for presence and, optionally, its checksum.

    Returns one :class:`FileCheck` per file instead of raising, so ``aurum
    doctor`` can report the full picture rather than the first problem.
    """
    directory = data_directory or DATA_DIRECTORY
    manifest = load_manifest(data_directory=directory)
    declared = {
        Path(key).name: value
        for key, value in dict(manifest.get("files", {})).items()
        if not key.startswith(RESERVED_PREFIX)
    }

    checks: list[FileCheck] = []
    for name in EXPECTED_FILES:
        path = directory / name
        if not path.is_file():
            checks.append(
                FileCheck(name, present=False, checksum_matches=None, detail="ausente")
            )
            continue
        expected = declared.get(name)
        if not verify_checksums or expected is None:
            detail = "" if expected else "sin checksum declarado"
            checks.append(
                FileCheck(name, present=True, checksum_matches=None, detail=detail)
            )
            continue
        actual = sha256_file(path)
        matches = actual == expected
        checks.append(
            FileCheck(
                name,
                present=True,
                checksum_matches=matches,
                detail=""
                if matches
                else f"esperado {expected[:12]}…, obtenido {actual[:12]}…",
            )
        )
    return checks


def clean_scalar(value: object) -> str:
    """Normalize one catalog cell into a trimmed string.

    Missing data becomes the empty string. This is the only place in the code
    base allowed to make that decision (P-07).
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    # pandas escribe los ausentes como "nan" al castear a str; el enunciado es
    # explícito en que eso NO es un valor.
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def clean_flag(value: object) -> bool:
    """Interpret a catalog boolean written by pandas, Excel, or by hand."""
    if isinstance(value, bool):
        return value
    text = clean_scalar(value).lower()
    if text in {"true", "1", "yes", "y", "si", "sí"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    raise DataIntegrityError(f"Valor booleano no reconocido: {value!r}")


def clean_version(value: object) -> int:
    """Interpret a catalog version, rejecting anything that is not a version."""
    text = clean_scalar(value)
    try:
        return int(float(text))
    except ValueError as error:
        raise DataIntegrityError(
            f"catalog_version no es un entero: {value!r}"
        ) from error


def expected_record_id(product_id: str) -> str:
    """Return the record_id that the manifest's ID contract demands.

    The catalog already ships a ``record_id``; this recomputes it so a silent
    mismatch —a re-exported file, a corrupted row— is caught before those IDs
    become point IDs in the vector database (RF-07).
    """
    resolved = clean_scalar(product_id)
    if not resolved:
        raise DataIntegrityError("product_id no puede estar vacío")
    return str(uuid.uuid5(_NAMESPACE, resolved))


def record_from_row(row: dict[str, Any]) -> CatalogRecord:
    """Build one sanitized record from a raw catalog row."""
    return CatalogRecord(
        record_id=clean_scalar(row["record_id"]),
        product_id=clean_scalar(row["product_id"]),
        title=clean_scalar(row["title"]),
        brand=clean_scalar(row["brand"]),
        color=clean_scalar(row["color"]),
        locale=clean_scalar(row["locale"]) or "es",
        text=clean_scalar(row["text"]),
        catalog_version=clean_version(row["catalog_version"]),
        active=clean_flag(row["active"]),
    )


def read_catalog_frame(
    profile: Profile = "sample", *, data_directory: Path | None = None
) -> pd.DataFrame:
    """Read one catalog profile as a raw frame, checking its columns."""
    if profile not in CATALOG_PROFILES:
        known = ", ".join(sorted(CATALOG_PROFILES))
        raise DataIntegrityError(f"Perfil desconocido {profile!r}. Usa uno de: {known}")
    directory = data_directory or DATA_DIRECTORY
    filename, _ = CATALOG_PROFILES[profile]
    path = directory / filename
    if not path.is_file():
        raise DataIntegrityError(f"No se encuentra el catálogo {path}")
    frame = pd.read_csv(path, dtype="object")
    missing = set(CATALOG_COLUMNS).difference(frame.columns)
    if missing:
        raise DataIntegrityError(f"Faltan columnas en {filename}: {sorted(missing)}")
    return frame


def load_catalog(
    profile: Profile = "sample",
    *,
    data_directory: Path | None = None,
    verify_ids: bool = True,
) -> CatalogSnapshot:
    """Load, sanitize, and validate one catalog profile.

    Validation happens here and only here, so that no layer downstream needs to
    wonder whether a brand might be ``"nan"`` or an ID might be duplicated.
    """
    directory = data_directory or DATA_DIRECTORY
    frame = read_catalog_frame(profile, data_directory=directory)
    records = tuple(
        record_from_row(row) for row in frame.to_dict(orient="records")
    )

    _validate_counts(records, profile, directory)
    _validate_uniqueness(records)
    if verify_ids:
        _validate_id_contract(records)

    return CatalogSnapshot(
        profile=profile,
        records=records,
        by_record_id={record.record_id: record for record in records},
        by_product_id={record.product_id: record for record in records},
    )


def _validate_counts(
    records: tuple[CatalogRecord, ...], profile: Profile, directory: Path
) -> None:
    manifest = load_manifest(data_directory=directory)
    _, counts_key = CATALOG_PROFILES[profile]
    expected = dict(manifest["counts"])[counts_key]
    if len(records) != expected:
        raise DataIntegrityError(
            f"El perfil {profile!r} declara {expected} registros en el manifiesto "
            f"pero se han cargado {len(records)}"
        )


def _validate_uniqueness(records: tuple[CatalogRecord, ...]) -> None:
    record_ids = {record.record_id for record in records}
    if len(record_ids) != len(records):
        raise DataIntegrityError(
            "Hay record_id duplicados: la ingesta perdería productos al hacer upsert"
        )
    product_ids = {record.product_id for record in records}
    if len(product_ids) != len(records):
        raise DataIntegrityError("Hay product_id duplicados en el catálogo")


def _validate_id_contract(records: tuple[CatalogRecord, ...]) -> None:
    for record in records:
        expected = expected_record_id(record.product_id)
        if record.record_id != expected:
            raise DataIntegrityError(
                f"El producto {record.product_id} declara record_id "
                f"{record.record_id} pero el contrato UUIDv5 exige {expected}"
            )
