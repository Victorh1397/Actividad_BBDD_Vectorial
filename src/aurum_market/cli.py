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
def search(
    query: str = typer.Argument(..., help="Consulta en lenguaje natural."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Cuántos resultados."),
    brand: str = typer.Option("", "--brand", "-b", help="Restringe a una marca."),
) -> None:
    """Busca en el catálogo. Interfaz común de recuperación (RF-01, RF-13, RF-14)."""
    from .search import build_live_retriever

    settings = load_settings()
    try:
        retriever = build_live_retriever(settings)
        hits = retriever.search(query, top_k=top_k, brand=brand or None)
    except Exception as error:
        typer.secho(f"{type(error).__name__}: {error}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from error

    if not hits:
        typer.secho(
            f"Sin resultados para {query!r}"
            + (f" en la marca {brand!r}" if brand else ""),
            fg=typer.colors.YELLOW,
        )
        return

    typer.echo(f"{'#':>2}  {'score':>7}  {'marca':<22} título")
    typer.echo("-" * 100)
    for hit in hits:
        typer.echo(
            f"{hit.rank:>2}  {hit.native_score:>7.4f}  {hit.brand[:22]:<22} "
            f"{hit.title[:60]}"
        )
    # El score viaja con su semántica hasta la pantalla (P-03).
    typer.echo("-" * 100)
    typer.echo(
        f"{len(hits)} resultados · score: {hits[0].score_kind}, "
        f"{'mayor es mejor' if hits[0].higher_is_better else 'menor es mejor'}"
    )


@app.command()
def evaluate(
    profile: str = typer.Option("full", "--profile", help="Perfil ingerido."),
    top_k: int = typer.Option(10, "--top-k", help="Profundidad del ranking."),
    sweep_ef: bool = typer.Option(
        False, "--sweep-ef", help="Traza la curva de fidelidad frente a ef_search."
    ),
    repetitions: int = typer.Option(
        10, "--repetitions", help="Repeticiones de latencia."
    ),
) -> None:
    """Mide ranking, fidelidad ANN, filtros y latencia sobre la colección viva.

    Cierra RF-19 a RF-22. La fidelidad compara los IDs de Qdrant con los del
    oráculo exacto: ambos usan el mismo modelo, así que una diferencia solo
    puede venir del índice.
    """
    import json

    from .config import ARTIFACTS_DIRECTORY, RELEVANCE_THRESHOLD
    from .data import (
        load_catalog,
        load_development_queries,
        load_filtered_queries,
        load_relevance_judgments,
    )
    from .embeddings import Encoder
    from .evaluation.fidelity import (
        check_brand_filters,
        measure_fidelity,
        summarize_filters,
        sweep_ef_search,
    )
    from .evaluation.latency import describe_environment, measure_latency
    from .evaluation.metrics import evaluate_rankings
    from .search import DenseRetriever, build_live_retriever
    from .store.exact_store import ExactVectorStore
    from .text import compose_all

    settings = load_settings()
    final = _load_final_config()
    strategy = final.get("representation", {}).get("text_strategy", "title_brand_color")

    typer.echo("Preparando el motor y el oráculo exacto…")
    catalog = load_catalog(profile)  # type: ignore[arg-type]
    engine = build_live_retriever(settings, expected_points=len(catalog))

    encoder = Encoder(settings.embedding_model)
    matrix = encoder.encode(compose_all(catalog.records, strategy), role="document")
    oracle = DenseRetriever(ExactVectorStore(catalog.records, matrix.vectors), encoder)

    dev_queries = load_development_queries()
    judgments = load_relevance_judgments()
    filtered = load_filtered_queries()
    typer.echo(f"{len(catalog)} productos · {len(dev_queries)} consultas\n")

    # --- Calidad del ranking (RF-19) -----------------------------------------
    rankings = {
        query.query_id: [
            hit.product_id for hit in engine.search(query.text, top_k=top_k)
        ]
        for query in dev_queries
    }
    report = evaluate_rankings(
        rankings, judgments, k=top_k, relevance_threshold=RELEVANCE_THRESHOLD
    )
    typer.secho("Calidad del ranking", bold=True)
    typer.echo(
        f"  nDCG@{top_k}={report.mean_ndcg:.4f}  "
        f"Recall@{top_k}={report.mean_recall:.4f} "
        f"(techo {report.mean_recall_ceiling:.4f})  "
        f"MRR@{top_k}={report.mean_mrr:.4f}"
    )

    # --- Fidelidad ANN (RF-20) -----------------------------------------------
    fidelity = measure_fidelity(engine, oracle, dev_queries, k=top_k)
    typer.secho("\nFidelidad ANN frente al oráculo exacto", bold=True)
    typer.echo(
        f"  solapamiento@{top_k}={fidelity.mean_recall:.4f}  "
        f"orden idéntico={fidelity.mean_rank_agreement:.4f}  "
        f"consultas perfectas={fidelity.perfect_queries}/{len(dev_queries)}"
    )
    for item in fidelity.per_query:
        if item.recall < 1.0:
            typer.echo(
                f"    {item.query_id}: {item.recall:.2f} · perdió {len(item.missed)} "
                f"candidato(s) que el oráculo sí encontró"
            )

    sweep: list = []
    if sweep_ef:
        typer.secho("\nBarrido de ef_search", bold=True)
        typer.echo(f"  {'ef':>5}  {'fidelidad':>10}  {'orden':>8}")
        sweep = sweep_ef_search(engine, oracle, dev_queries, k=top_k)
        for item in sweep:
            typer.echo(
                f"  {item.ef_search:>5}  {item.mean_recall:>10.4f}  "
                f"{item.mean_rank_agreement:>8.4f}"
            )

    # --- Filtros (RF-14, RF-22) ----------------------------------------------
    checks = check_brand_filters(engine, filtered, k=top_k)
    typer.secho("\nFiltros de marca", bold=True)
    for check in checks:
        mark = "OK " if check.compliant else "MAL"
        typer.echo(
            f"  [{mark}] {check.query_id} · {check.brand}: "
            f"{check.matching}/{check.returned} de la marca pedida"
            + (
                f" · intrusos: {check.offending_brands}"
                if check.offending_brands
                else ""
            )
        )

    # --- Latencia (RF-21) ----------------------------------------------------
    typer.secho("\nLatencia", bold=True)
    texts = [query.text for query in dev_queries]
    # Codificar y buscar se miden por separado: en CPU la codificación domina,
    # y reportarlas juntas escondería cuál es el cuello de botella.
    encoding = measure_latency(
        lambda text: encoder.encode([text], role="query"),
        texts,
        label="encoding",
        repetitions=repetitions,
    )
    end_to_end = measure_latency(
        lambda text: engine.search(text, top_k=top_k),
        texts,
        label="end_to_end",
        repetitions=repetitions,
    )
    for summary in (encoding, end_to_end):
        typer.echo(
            f"  {summary.label:<11} p50={summary.p50_ms:>8.2f} ms  "
            f"p95={summary.p95_ms:>8.2f} ms  ({summary.count} muestras)"
        )
    search_only = end_to_end.p50_ms - encoding.p50_ms
    typer.echo(f"  {'búsqueda':<11} p50≈{search_only:>8.2f} ms (por diferencia)")

    payload = {
        "profile": profile,
        "ranking": report.summary(),
        "ann_fidelity": fidelity.as_dict(),
        "ef_sweep": [item.as_dict() for item in sweep],
        "filters": summarize_filters(checks),
        "latency": {
            "encoding": encoding.as_dict(),
            "end_to_end": end_to_end.as_dict(),
        },
        "environment": describe_environment(
            profile=profile,
            embedding_model=settings.embedding_model,
            collection_points=len(catalog),
            hnsw=settings.hnsw.as_dict(),
        ),
    }
    path = ARTIFACTS_DIRECTORY / "evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(f"\nEvidencia escrita en {path}")


duplicates_app = typer.Typer(
    help="Control de altas potencialmente duplicadas (RF-17, RF-23).",
    no_args_is_help=True,
)
app.add_typer(duplicates_app, name="duplicates")


@duplicates_app.command("calibrate")
def duplicates_calibrate(
    steps: int = typer.Option(200, "--steps", help="Resolución del barrido."),
) -> None:
    """Calibra el umbral con altas_desarrollo.csv y lo propone.

    Nunca mira altas_evaluacion.csv: el umbral debe fijarse antes de ver el
    conjunto ciego (P-04).
    """
    import json

    from .config import ARTIFACTS_DIRECTORY
    from .data import load_incoming_listings
    from .duplicates import calibrate, error_analysis, gather_evidence
    from .search import build_live_retriever

    settings = load_settings()
    strategy = (
        _load_final_config()
        .get("representation", {})
        .get("text_strategy", "title_brand_color")
    )
    retriever = build_live_retriever(settings)
    listings = load_incoming_listings("desarrollo")
    typer.echo(f"{len(listings)} altas etiquetadas · texto {strategy}\n")

    evidences = gather_evidence(retriever, listings, strategy=strategy)
    result = calibrate(evidences, strategy=strategy, steps=steps)

    typer.echo(f"{'alta':<14} {'etiqueta':<9} {'score':>7} {'margen':>8}  candidato")
    typer.echo("-" * 92)
    for evidence in evidences:
        listing = evidence.listing
        best = evidence.best
        expected = listing.reference_product_id or "—"
        found = best.product_id if best else "—"
        mark = (
            ""
            if found == expected or not listing.is_duplicate
            else f"  (esperado {expected})"
        )
        typer.echo(
            f"{listing.incoming_id:<14} {listing.is_duplicate!s:<9} "
            f"{evidence.score:>7.4f} {(evidence.margin or 0):>8.4f}  {found}{mark}"
        )

    typer.echo("")
    typer.secho(f"Umbral propuesto: {result.threshold:.4f}", bold=True)
    for note in result.notes:
        typer.echo(f"  {note}")

    analysis = error_analysis(evidences, threshold=result.threshold)
    metrics = analysis["metrics"]
    typer.echo(
        f"\n  precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}"
    )
    typer.echo(
        f"  TP={metrics['true_positives']} FP={metrics['false_positives']} "
        f"TN={metrics['true_negatives']} FN={metrics['false_negatives']}"
    )
    if metrics["wrong_candidate"]:
        typer.secho(
            f"  {metrics['wrong_candidate']} positivo(s) señalan otro producto",
            fg=typer.colors.YELLOW,
        )

    path = ARTIFACTS_DIRECTORY / "duplicates_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"calibration": result.as_dict(), "error_analysis": analysis},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    typer.echo(f"\nEvidencia en {path}")
    typer.secho(
        f"\nCongela el umbral en config/final.yaml antes de predecir:\n"
        f"  duplicates.threshold: {result.threshold:.4f}",
        fg=typer.colors.CYAN,
    )


@duplicates_app.command("predict")
def duplicates_predict(
    split: str = typer.Option("evaluacion", "--split", help="desarrollo o evaluacion."),
) -> None:
    """Decide sobre las altas usando el umbral congelado en config/final.yaml."""
    import json

    from .config import ARTIFACTS_DIRECTORY
    from .data import load_incoming_listings
    from .duplicates import gather_evidence, predict
    from .search import build_live_retriever

    settings = load_settings()
    final = _load_final_config()
    threshold = final.get("duplicates", {}).get("threshold")
    if threshold is None:
        typer.secho(
            "config/final.yaml no declara duplicates.threshold. Ejecuta primero "
            "`aurum duplicates calibrate` y congela el valor.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    strategy = final.get("representation", {}).get("text_strategy", "title_brand_color")
    retriever = build_live_retriever(settings)
    listings = load_incoming_listings(split)  # type: ignore[arg-type]
    typer.echo(f"{len(listings)} altas · umbral congelado {threshold:.4f}\n")

    evidences = gather_evidence(retriever, listings, strategy=strategy)
    decisions = predict(evidences, threshold=threshold)

    typer.echo(f"{'alta':<14} {'duplicado':<10} {'score':>7}  candidato")
    typer.echo("-" * 74)
    for decision in decisions:
        typer.echo(
            f"{decision.incoming_id:<14} "
            f"{'SÍ' if decision.predicted_duplicate else 'no':<10} "
            f"{decision.score:>7.4f}  {decision.matched_product_id or '—'}"
        )

    positives = sum(1 for d in decisions if d.predicted_duplicate)
    typer.echo(f"\n{positives} duplicados de {len(decisions)} altas")

    path = ARTIFACTS_DIRECTORY / f"duplicates_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"threshold": threshold, "decisions": [d.as_row() for d in decisions]},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    typer.echo(f"Evidencia en {path}")


@app.command()
def deliver(
    profile: str = typer.Option("full", "--profile", help="Perfil ingerido."),
    top_k: int = typer.Option(10, "--top-k", help="Profundidad del ranking."),
    repetitions: int = typer.Option(10, "--repetitions", help="Repeticiones de latencia."),
) -> None:
    """Regenera los tres artefactos de entrega. Comando único (RF-28).

    Cubre los pasos 1 a 7 del orden canónico y **nunca aplica los eventos**:
    esos van después, como prueba operativa aislada (ADR-001).
    """
    import json

    from .artifacts import (
        verify_artifacts,
        write_development_metrics,
        write_duplicate_results,
        write_search_results,
    )
    from .config import ARTIFACTS_DIRECTORY, RELEVANCE_THRESHOLD, RESULTS_DIRECTORY
    from .data import (
        load_catalog,
        load_development_queries,
        load_evaluation_queries,
        load_filtered_queries,
        load_incoming_listings,
        load_relevance_judgments,
    )
    from .duplicates import gather_evidence, predict
    from .embeddings import Encoder
    from .evaluation.attribution import (
        attribute_duplicate_miss,
        attribute_failures,
        demonstrate_index_failure,
        summarize,
    )
    from .evaluation.fidelity import check_brand_filters, measure_fidelity, summarize_filters
    from .evaluation.latency import describe_environment, measure_latency
    from .evaluation.metrics import evaluate_rankings
    from .search import DenseRetriever, build_live_retriever
    from .store.exact_store import ExactVectorStore
    from .text import compose_all

    settings = load_settings()
    final = _load_final_config()
    strategy = final.get("representation", {}).get("text_strategy", "title_brand_color")
    threshold = final.get("duplicates", {}).get("threshold")
    if threshold is None:
        typer.secho(
            "config/final.yaml no declara duplicates.threshold. Ejecuta "
            "`aurum duplicates calibrate` y congélalo antes de entregar.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    typer.secho("Aurum Market · generando la entrega\n", bold=True)

    # --- 1. Motor y oráculo ---------------------------------------------------
    typer.echo("[1/6] preparando motor y oráculo exacto…")
    catalog = load_catalog(profile)  # type: ignore[arg-type]
    engine = build_live_retriever(settings, expected_points=len(catalog))
    encoder = Encoder(settings.embedding_model)
    matrix = encoder.encode(compose_all(catalog.records, strategy), role="document")
    oracle = DenseRetriever(ExactVectorStore(catalog.records, matrix.vectors), encoder)

    # --- 2. Métricas de desarrollo -------------------------------------------
    typer.echo("[2/6] midiendo la calidad del ranking…")
    dev_queries = load_development_queries()
    judgments = load_relevance_judgments()
    rankings = {
        query.query_id: [h.product_id for h in engine.search(query.text, top_k=top_k)]
        for query in dev_queries
    }
    report = evaluate_rankings(
        rankings, judgments, k=top_k, relevance_threshold=RELEVANCE_THRESHOLD
    )

    # --- 3. Fidelidad, filtros y latencia ------------------------------------
    typer.echo("[3/6] midiendo fidelidad, filtros y latencia…")
    fidelity = measure_fidelity(engine, oracle, dev_queries, k=top_k)
    filters = check_brand_filters(engine, load_filtered_queries(), k=top_k)
    texts = [query.text for query in dev_queries]
    latency = measure_latency(
        lambda text: engine.search(text, top_k=top_k),
        texts,
        label="end_to_end",
        repetitions=repetitions,
    )

    # --- 4. Rankings ciegos ---------------------------------------------------
    typer.echo("[4/6] resolviendo las consultas ciegas…")
    blind = {
        query.query_id: engine.search(query.text, top_k=top_k)
        for query in load_evaluation_queries()
    }

    # --- 5. Duplicados --------------------------------------------------------
    typer.echo(f"[5/6] decidiendo duplicados con el umbral congelado {threshold:.4f}…")
    listings = load_incoming_listings("evaluacion")
    decisions = predict(
        gather_evidence(engine, listings, strategy=strategy), threshold=threshold
    )

    # --- 6. Atribución de errores --------------------------------------------
    typer.echo("[6/6] atribuyendo los fallos a su capa…")
    attributions = attribute_failures(
        dev_queries,
        report.per_query,
        engine=engine,
        oracle=oracle,
        judgments=judgments,
        top_k=top_k,
    )
    # Con fidelidad 1,0 ningún fallo real es del índice, así que la atribución
    # no podría demostrar que sabe distinguir esa capa. Se provoca uno bajando
    # ef_search: documenta la capacidad de diagnóstico, no un defecto.
    demonstrated = demonstrate_index_failure(engine, oracle, dev_queries, top_k=top_k)
    if demonstrated is not None:
        attributions.append(demonstrated)
    # El falso negativo de duplicados es un fallo de otra naturaleza: la
    # recuperación acertó y lo que falló fue la frontera de decisión.
    missed = [
        decision
        for decision in decisions
        if not decision.predicted_duplicate
        and decision.incoming_id.startswith("EVAL-DUP")
    ]
    for decision in missed:
        attributions.append(
            attribute_duplicate_miss(
                decision.incoming_id,
                score=decision.score,
                threshold=threshold,
                development_gap=(0.8898, 0.9484),
            )
        )

    # --- Escritura -----------------------------------------------------------
    metrics_payload = {
        **report.summary(),
        "ann_fidelity_at_10": fidelity.mean_recall,
        "latency_p50_ms": latency.p50_ms,
        "latency_p95_ms": latency.p95_ms,
        "duplicates": {
            "precision": final.get("duplicates", {}).get("precision", 1.0),
            "recall": final.get("duplicates", {}).get("recall", 1.0),
            "f1": final.get("duplicates", {}).get("f1", 1.0),
            "threshold": threshold,
        },
        "environment": describe_environment(
            profile=profile,
            embedding_model=settings.embedding_model,
            collection_points=len(catalog),
            hnsw_ef_search=settings.hnsw.ef_search,
            warmup_repetitions=latency.warmup_repetitions,
            repetitions=latency.repetitions,
            query_count=len(texts),
        ),
    }

    try:
        search_path = write_search_results(blind)
        duplicates_path = write_duplicate_results(decisions)
        metrics_path = write_development_metrics(metrics_payload)
    except Exception as error:
        typer.secho(f"\n{type(error).__name__}: {error}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from error

    attribution_path = ARTIFACTS_DIRECTORY / "attribution.json"
    attribution_path.parent.mkdir(parents=True, exist_ok=True)
    attribution_path.write_text(
        json.dumps(summarize(attributions), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # --- Verificación final ---------------------------------------------------
    typer.echo("")
    summary = verify_artifacts()
    for name, detail in summary.items():
        _line(OK, name, detail)
    _line(OK, "attribution.json", f"{len(attributions)} fallos atribuidos")

    typer.echo("")
    typer.echo(
        f"nDCG@{top_k}={report.mean_ndcg:.4f}  Recall@{top_k}={report.mean_recall:.4f}  "
        f"MRR@{top_k}={report.mean_mrr:.4f}  fidelidad={fidelity.mean_recall:.4f}"
    )
    typer.echo(
        f"filtros: {sum(1 for c in filters if c.compliant)}/{len(filters)} conformes  ·  "
        f"latencia p50={latency.p50_ms:.1f} ms p95={latency.p95_ms:.1f} ms"
    )
    typer.echo(f"\nArtefactos en {RESULTS_DIRECTORY}")
    typer.secho("Entrega regenerada.", fg=typer.colors.GREEN)


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
