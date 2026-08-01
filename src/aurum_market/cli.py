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
def experiment(
    profile: str = typer.Option(
        "sample",
        "--profile",
        help="Perfil de catálogo: sample (1.500) o full (15.000).",
    ),
    top_k: int = typer.Option(10, "--top-k", help="Profundidad del ranking."),
    only: str = typer.Option(
        "", "--only", help="Ejecuta solo estos experimentos, separados por comas."
    ),
) -> None:
    """Compara representaciones sobre el conjunto de desarrollo (RF-06).

    Todos los experimentos usan el oráculo exacto, nunca un índice ANN: así una
    diferencia solo puede venir de la representación. La pérdida del índice se
    mide aparte.
    """
    from .data import (
        load_catalog,
        load_development_queries,
        load_relevance_judgments,
    )
    from .experiments import (
        build_matrix,
        comparison_table,
        run_experiment,
        write_result,
    )

    if profile not in ("sample", "full"):
        typer.secho(f"Perfil desconocido: {profile!r}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Cargando el catálogo ({profile})…")
    catalog = load_catalog(profile)  # type: ignore[arg-type]
    queries = load_development_queries()
    judgments = load_relevance_judgments()
    typer.echo(
        f"{len(catalog)} productos · {len(queries)} consultas · "
        f"{sum(len(v) for v in judgments.values())} juicios\n"
    )

    selected = {item.strip().upper() for item in only.split(",") if item.strip()}
    results = []
    winner_strategy = None

    for config in build_matrix():
        if selected and config.experiment_id not in selected:
            continue
        # E3 hereda la estrategia ganadora: su única variable es el modelo.
        if config.experiment_id == "E3" and winner_strategy is not None:
            config = build_matrix(winner_strategy=winner_strategy)[3]

        typer.echo(f"[{config.experiment_id}] {config.description}")
        try:
            result = run_experiment(
                config,
                catalog.records,
                queries,
                judgments,
                profile=profile,  # type: ignore[arg-type]
                top_k=top_k,
                show_progress=True,
            )
        except Exception as error:  # el motivo importa más que el traceback
            typer.secho(
                f"  falló: {type(error).__name__}: {error}", fg=typer.colors.RED
            )
            continue

        path = write_result(result)
        typer.echo(
            f"  nDCG@{top_k}={result.report.mean_ndcg:.4f}  "
            f"Recall@{top_k}={result.report.mean_recall:.4f}  "
            f"MRR@{top_k}={result.report.mean_mrr:.4f}   → {path.name}\n"
        )
        results.append(result)

        if config.experiment_id == "E2" and len(results) >= 2:
            previous = next(
                (r for r in results if r.config.experiment_id == "E1"), None
            )
            if previous is not None:
                winner = max((previous, result), key=lambda r: r.report.mean_ndcg)
                winner_strategy = winner.config.text_strategy

    if not results:
        typer.secho("No se ejecutó ningún experimento.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    typer.echo(comparison_table(results))


def _load_final_config() -> dict:
    """Read config/final.yaml, the frozen configuration of the final run."""
    import yaml

    path = PROJECT_ROOT / "config" / "final.yaml"
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@app.command()
def ingest(
    profile: str = typer.Option(
        "full", "--profile", help="Perfil de catálogo: sample (1.500) o full (15.000)."
    ),
    batch_size: int = typer.Option(0, "--batch-size", help="0 usa el valor del .env."),
) -> None:
    """Ingiere el catálogo en Qdrant, por lotes e idempotente (RF-09, RF-10).

    Repetirla no aumenta el recuento: cada producto se escribe bajo su UUIDv5.
    """
    from .data import load_catalog
    from .embeddings import Encoder
    from .ingest import ingest_catalog
    from .store.qdrant_store import QdrantStore

    if profile not in ("sample", "full"):
        typer.secho(f"Perfil desconocido: {profile!r}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        settings = load_settings()
    except ConfigurationError as error:
        typer.secho(str(error), fg=typer.colors.RED)
        raise typer.Exit(code=1) from error

    final = _load_final_config()
    strategy = final.get("representation", {}).get("text_strategy", "title_brand_color")

    typer.echo(f"Cargando el catálogo ({profile})…")
    catalog = load_catalog(profile)  # type: ignore[arg-type]
    typer.echo(
        f"{len(catalog)} productos · modelo {settings.embedding_model} · "
        f"texto {strategy}\n"
    )

    store = QdrantStore(settings)
    encoder = Encoder(settings.embedding_model, batch_size=64)

    before = store.status()
    if before.exists:
        typer.echo(f"La colección ya existe con {before.points_count} puntos.")

    with typer.progressbar(length=len(catalog), label="ingiriendo") as bar:
        seen = 0

        def advance(sent: int, _total: int) -> None:
            nonlocal seen
            bar.update(sent - seen)
            seen = sent

        try:
            report = ingest_catalog(
                catalog,
                store,
                encoder,
                text_strategy=strategy,
                batch_size=batch_size or settings.batch_size,
                show_progress=True,
                on_batch=advance,
            )
        except Exception as error:
            typer.secho(f"\n{type(error).__name__}: {error}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from error

    typer.echo("")
    typer.echo(f"lotes enviados      : {report.batches}")
    typer.echo(f"puntos en la colección: {report.points_count}")
    typer.echo(f"vectores indexados  : {report.status.indexed_vectors_count}")
    typer.echo(f"dimensión           : {report.dimension}")
    if before.exists:
        delta = report.points_count - before.points_count
        typer.echo(f"variación del recuento: {delta:+d}")
    typer.secho("Ingesta completada.", fg=typer.colors.GREEN)


@app.command()
def verify(
    profile: str = typer.Option("full", "--profile", help="Perfil esperado."),
) -> None:
    """Comprueba recuento, dimensión y estado antes de aceptar consultas (RF-10)."""
    from .data import CATALOG_PROFILES, load_manifest
    from .ingest import verify_collection
    from .store.qdrant_store import QdrantStore

    settings = load_settings()
    expected = dict(load_manifest()["counts"])[CATALOG_PROFILES[profile][1]]
    store = QdrantStore(settings)

    try:
        status = verify_collection(
            store,
            expected_points=int(expected),
            expected_dimension=settings.embedding_dimension,
        )
    except Exception as error:
        typer.secho(f"[ FALLO] {error}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from error

    _line(OK, f"Colección {store.collection}")
    _line(OK, f"{status.points_count} puntos (esperados {expected})")
    _line(OK, f"dimensión {status.dimension} · distancia {status.distance}")
    # Leído del motor, no de nuestra configuración: declarar y aplicar son
    # cosas distintas, y RF-08 pide comprobar la segunda.
    _line(
        OK,
        f"HNSW aplicado: m={status.hnsw_m}, ef_construct={status.hnsw_ef_construct}",
    )
    if status.fully_indexed:
        _line(
            OK,
            f"{status.indexed_vectors_count} vectores indexados en "
            f"{status.segments_count} segmentos",
        )
    else:
        _line(
            WARN,
            f"{status.indexed_vectors_count} de {status.points_count} indexados",
            f"~{status.kilobytes_per_segment:.0f} KB por segmento frente al umbral "
            f"de {status.indexing_threshold} KB. Por debajo, Qdrant responde por "
            "fuerza bruta y el grafo HNSW no interviene",
        )
    typer.secho("La colección puede aceptar consultas.", fg=typer.colors.GREEN)


@app.command()
def reset() -> None:
    """Borra la colección. Destructivo y desactivado por defecto (RF-18)."""
    from .store.qdrant_store import QdrantStore

    settings = load_settings()
    store = QdrantStore(settings)

    if not settings.cleanup_authorized(store.collection):
        typer.secho(
            f"Operación bloqueada sobre {store.collection!r}.", fg=typer.colors.RED
        )
        typer.echo("Requiere las DOS condiciones en el .env:")
        typer.echo("  AURUM_ALLOW_RESET=true")
        typer.echo(f"  AURUM_CONFIRM_CLEANUP={store.collection}")
        raise typer.Exit(code=1)

    status = store.status()
    store.reset()
    typer.secho(
        f"Colección {store.collection} eliminada ({status.points_count} puntos).",
        fg=typer.colors.YELLOW,
    )


@app.command()
def version() -> None:
    """Muestra la versión del paquete y la ruta del proyecto."""
    from . import __version__

    typer.echo(f"aurum-market {__version__}")
    typer.echo(f"proyecto: {Path(PROJECT_ROOT)}")


if __name__ == "__main__":  # pragma: no cover
    app()
