"""The representation experiment matrix (RF-06).

The statement is explicit: *"Comparad al menos dos configuraciones relevantes.
Cambiar solo el nombre del modelo sin analizar el resultado no constituye un
experimento"*. So every run here isolates **one** variable against a named
comparison, and records configuration, metrics and retrieved IDs — without the
three, a number is an anecdote (P-06).

All runs use the exact NumPy oracle, never an ANN index. Mixing both would
leave any difference ambiguous between the representation and the index; the
index is measured separately in RF-20.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .baselines import TfidfRetriever
from .config import ARTIFACTS_DIRECTORY, CONTRACTS_DIRECTORY, RELEVANCE_THRESHOLD
from .contracts import CatalogRecord, Profile, RetrievalQuery
from .embeddings import Encoder
from .evaluation.metrics import EvaluationReport, evaluate_rankings
from .search import DenseRetriever, Retriever
from .store.exact_store import ExactVectorStore
from .text import TextStrategy, compose_all, describe

DEFAULT_TOP_K = 10


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """One point in the comparison matrix."""

    experiment_id: str
    retriever: str
    text_strategy: TextStrategy
    embedding_model: str | None = None
    compares_with: str | None = None
    isolates: str = ""

    @property
    def description(self) -> str:
        if self.compares_with is None:
            return f"Referencia inicial. {self.isolates}"
        return f"Frente a {self.compares_with}: aísla {self.isolates}"


@dataclass(slots=True)
class ExperimentResult:
    """An auditable run: configuration, metrics and the IDs behind them."""

    config: ExperimentConfig
    report: EvaluationReport
    retrieved_ids: dict[str, list[str]]
    profile: Profile
    catalog_size: int
    notes: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def as_dict(self) -> dict[str, object]:
        """Serialise into the shape declared by experiment_run.schema.json."""
        return {
            "experiment_id": self.config.experiment_id,
            "description": self.config.description,
            "configuration": {
                "retriever": self.config.retriever,
                "embedding_model": self.config.embedding_model,
                "text_strategy": self.config.text_strategy,
                "metric": "tfidf_cosine"
                if self.config.retriever == "tfidf"
                else "cosine",
                "normalized": True,
                "profile": self.profile,
                "top_k": self.report.k,
                "relevance_threshold": self.report.relevance_threshold,
                "hnsw": None,
            },
            "metrics": {
                # Sin sufijo @k: la profundidad se declara una sola vez, en
                # configuration.top_k. Duplicarla en el nombre del campo obliga
                # a un contrato distinto por cada k.
                "ndcg": self.report.mean_ndcg,
                "recall": self.report.mean_recall,
                "mrr": self.report.mean_mrr,
                "recall_ceiling": self.report.mean_recall_ceiling,
                "per_query": self.report.per_query_rows(),
            },
            "retrieved_ids": self.retrieved_ids,
            "notes": self.notes,
            "generated_at": self.generated_at,
        }


def build_matrix(
    *, winner_strategy: TextStrategy | None = None
) -> tuple[ExperimentConfig, ...]:
    """Return the four planned runs.

    ``E3`` reuses whichever text strategy won between E1 and E2, so the only
    variable it changes is the model size.
    """
    return (
        ExperimentConfig(
            experiment_id="E0",
            retriever="tfidf",
            text_strategy="raw_text",
            isolates="el baseline léxico que el sistema denso debe superar",
        ),
        ExperimentConfig(
            experiment_id="E1",
            retriever="dense_exact",
            text_strategy="raw_text",
            embedding_model="intfloat/multilingual-e5-small",
            compares_with="E0",
            isolates="el efecto de pasar de coincidencia léxica a semántica",
        ),
        ExperimentConfig(
            experiment_id="E2",
            retriever="dense_exact",
            text_strategy="title_brand_color",
            embedding_model="intfloat/multilingual-e5-small",
            compares_with="E1",
            isolates="el efecto de qué texto se codifica, con el mismo modelo",
        ),
        ExperimentConfig(
            experiment_id="E3",
            retriever="dense_exact",
            text_strategy=winner_strategy or "title_brand_color",
            embedding_model="intfloat/multilingual-e5-base",
            compares_with="E2",
            isolates="el efecto del tamaño del modelo, con el mismo texto",
        ),
    )


def build_retriever(
    config: ExperimentConfig,
    records: Sequence[CatalogRecord],
    *,
    show_progress: bool = False,
) -> Retriever:
    """Construct the retriever a configuration describes."""
    if config.retriever == "tfidf":
        return TfidfRetriever(records, strategy=config.text_strategy)
    if config.retriever != "dense_exact":
        raise ValueError(f"Recuperador desconocido: {config.retriever!r}")
    if not config.embedding_model:
        raise ValueError(f"{config.experiment_id} necesita declarar un modelo")

    encoder = Encoder(config.embedding_model)
    texts = compose_all(records, config.text_strategy)
    matrix = encoder.encode(texts, role="document", show_progress=show_progress)
    store = ExactVectorStore(records, matrix.vectors)
    return DenseRetriever(store, encoder)


def run_experiment(
    config: ExperimentConfig,
    records: Sequence[CatalogRecord],
    queries: Sequence[RetrievalQuery],
    judgments: Mapping[str, Mapping[str, float]],
    *,
    profile: Profile = "sample",
    top_k: int = DEFAULT_TOP_K,
    show_progress: bool = False,
) -> ExperimentResult:
    """Run one configuration over the development workload and measure it."""
    retriever = build_retriever(config, records, show_progress=show_progress)

    retrieved: dict[str, list[str]] = {}
    for query in queries:
        hits = retriever.search(query.text, top_k=top_k)
        retrieved[query.query_id] = [hit.product_id for hit in hits]

    report = evaluate_rankings(
        retrieved, judgments, k=top_k, relevance_threshold=RELEVANCE_THRESHOLD
    )
    notes = [
        describe(config.text_strategy),
        f"Perfil {profile} con {len(records)} productos. La muestra contiene los "
        "248 productos juzgados, así que los relevantes son los mismos que en el "
        "catálogo completo; hay menos distractores, de modo que las cifras "
        "absolutas son optimistas y solo deben usarse para comparar entre sí.",
    ]
    return ExperimentResult(
        config=config,
        report=report,
        retrieved_ids=retrieved,
        profile=profile,
        catalog_size=len(records),
        notes=notes,
    )


def write_result(result: ExperimentResult, *, directory: Path | None = None) -> Path:
    """Persist one run, validating it against its declared contract."""
    payload = result.as_dict()
    _validate_against_contract(payload)
    target = directory or ARTIFACTS_DIRECTORY / "experiments"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{result.config.experiment_id}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _validate_against_contract(payload: Mapping[str, object]) -> None:
    from jsonschema import Draft202012Validator

    schema_path = CONTRACTS_DIRECTORY / "experiment_run.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)


def comparison_table(results: Sequence[ExperimentResult]) -> str:
    """Render the compact comparative table the statement asks for."""
    header = (
        f"{'exp':<4} {'recuperador':<12} {'texto':<18} {'modelo':<24} "
        f"{'nDCG@10':>8} {'Recall@10':>10} {'MRR@10':>8}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        model = result.config.embedding_model or "—"
        lines.append(
            f"{result.config.experiment_id:<4} "
            f"{result.config.retriever:<12} "
            f"{result.config.text_strategy:<18} "
            f"{model.split('/')[-1]:<24} "
            f"{result.report.mean_ndcg:>8.4f} "
            f"{result.report.mean_recall:>10.4f} "
            f"{result.report.mean_mrr:>8.4f}"
        )
    if results:
        ceiling = results[0].report.mean_recall_ceiling
        lines.append("-" * len(header))
        lines.append(f"{'':<4} techo estructural de Recall@10: {ceiling:.4f}")
    return "\n".join(lines)
