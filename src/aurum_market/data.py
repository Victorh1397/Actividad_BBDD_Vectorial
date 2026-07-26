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
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIRECTORY

# Los ficheros bajo datos/profesorado/ del manifiesto son las relevancias y
# duplicados reservados a la corrección: no forman parte de la entrega y su
# ausencia es lo correcto.
RESERVED_PREFIX = "datos/profesorado/"

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
