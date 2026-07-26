"""Command line interface for Aurum Market.

``aurum deliver`` is the single command that regenerates every deliverable
artifact (RF-28). Every other command exists to make one step observable on its
own.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import typer

from .config import PROJECT_ROOT, ConfigurationError, Settings, load_settings
from .data import DataIntegrityError, load_manifest, verify_data_integrity

app = typer.Typer(
    name="aurum",
    help="Aurum Market · búsqueda semántica y control de catálogo.",
    no_args_is_help=True,
    add_completion=False,
)

OK = "  OK  "
FAIL = " FALLO"
WARN = " AVISO"


def _line(status: str, message: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    typer.echo(f"[{status}] {message}{suffix}")


def _check_python() -> bool:
    version = sys.version_info
    supported = version.major == 3 and version.minor == 12
    message = f"Python {platform.python_version()}"
    if supported:
        _line(OK, message)
        return True
    _line(
        WARN,
        message,
        "el proyecto fija 3.12 en .python-version; usa `uv run` para que uv "
        "resuelva el intérprete correcto",
    )
    return True


def _check_env_file() -> bool:
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        _line(OK, ".env presente")
        return True
    _line(FAIL, ".env ausente", "ejecuta `make setup` o copia .env.example a .env")
    return False


def _check_settings() -> Settings | None:
    try:
        settings = load_settings()
    except ConfigurationError as error:
        _line(FAIL, "Configuración inválida", str(error))
        return None
    _line(OK, f"Modelo de embeddings: {settings.embedding_model}")
    _line(OK, f"Dimensión declarada: {settings.embedding_dimension}")
    _line(OK, f"Colección: {settings.qdrant_collection}")
    _line(
        OK,
        "HNSW: "
        f"m={settings.hnsw.m}, ef_construct={settings.hnsw.ef_construct}, "
        f"ef_search={settings.hnsw.ef_search}",
    )
    return settings


def _check_data(*, verify_checksums: bool) -> bool:
    try:
        manifest = load_manifest()
        checks = verify_data_integrity(verify_checksums=verify_checksums)
    except DataIntegrityError as error:
        _line(FAIL, "Datos", str(error))
        return False

    snapshot = manifest.get("snapshot_id", "desconocido")
    _line(OK, f"Manifiesto: {snapshot}")

    healthy = True
    for check in checks:
        if not check.present:
            _line(FAIL, check.name, check.detail)
            healthy = False
        elif check.checksum_matches is False:
            _line(FAIL, f"{check.name} (checksum)", check.detail)
            healthy = False
        elif check.checksum_matches is None and check.detail:
            _line(WARN, check.name, check.detail)
        else:
            _line(OK, check.name)
    return healthy


def _check_qdrant(settings: Settings) -> bool:
    try:
        from qdrant_client import QdrantClient
    except ImportError:  # pragma: no cover - dependencia declarada
        _line(FAIL, "qdrant-client no instalado", "ejecuta `make setup`")
        return False

    try:
        client = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5
        )
        collections = [item.name for item in client.get_collections().collections]
    except Exception as error:  # queremos el motivo exacto en pantalla
        _line(
            FAIL,
            f"Qdrant en {settings.qdrant_url}",
            f"{type(error).__name__}: {error}. Arranca Docker Desktop y ejecuta `make up`",
        )
        return False

    _line(OK, f"Qdrant accesible en {settings.qdrant_url}")
    if settings.qdrant_collection in collections:
        info = client.get_collection(settings.qdrant_collection)
        _line(
            OK,
            f"Colección {settings.qdrant_collection}",
            f"{info.points_count} puntos, {info.indexed_vectors_count} vectores indexados",
        )
    else:
        _line(
            WARN,
            f"Colección {settings.qdrant_collection} todavía no existe",
            "se creará con `aurum ingest`",
        )
    return True


@app.command()
def doctor(
    skip_checksums: bool = typer.Option(
        False,
        "--skip-checksums",
        help="Omite el cálculo de SHA-256 (más rápido, menos garantías).",
    ),
    skip_qdrant: bool = typer.Option(
        False, "--skip-qdrant", help="No intenta conectar con el motor vectorial."
    ),
) -> None:
    """Verifica entorno, configuración, datos y conectividad. No modifica nada."""
    typer.echo("Aurum Market · diagnóstico del entorno\n")

    healthy = _check_python()
    healthy &= _check_env_file()

    settings = _check_settings()
    healthy &= settings is not None

    typer.echo("")
    healthy &= _check_data(verify_checksums=not skip_checksums)

    if settings is not None and not skip_qdrant:
        typer.echo("")
        healthy &= _check_qdrant(settings)

    typer.echo("")
    if healthy:
        typer.secho("Entorno listo.", fg=typer.colors.GREEN)
        return
    typer.secho(
        "El entorno no está listo. Revisa los FALLO anteriores.", fg=typer.colors.RED
    )
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Muestra la versión del paquete y la ruta del proyecto."""
    from . import __version__

    typer.echo(f"aurum-market {__version__}")
    typer.echo(f"proyecto: {Path(PROJECT_ROOT)}")


if __name__ == "__main__":  # pragma: no cover
    app()
