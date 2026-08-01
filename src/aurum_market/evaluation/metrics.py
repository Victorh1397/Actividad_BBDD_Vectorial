"""Ranking metrics over graded relevance judgments (RF-19).

Adapted from ``evaluation.py`` of session 01, narrowed to the three metrics the
statement requires and to one identifier: ``product_id``.

Two decisions are fixed here and must not drift (P-05):

* Graded relevance is ``E=3, S=2, C=1, I=0``, set by the statement.
* "Relevant" for Recall and MRR means ``relevance >= 2``, i.e. Exact and
  Substitute. See specs/decisiones/ADR-004-umbral-de-relevancia.md.

nDCG uses the whole scale, so a Complement still contributes gain there.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from ..config import RELEVANCE_THRESHOLD

GradedQrels = Mapping[str, float]


class MetricError(ValueError):
    """Raised when a ranking or a set of judgments cannot be evaluated."""


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """The three required metrics for one query, plus the counts behind them."""

    query_id: str
    k: int
    ndcg: float
    recall: float
    mrr: float
    relevant_total: int
    relevant_retrieved: int
    retrieved: int

    @property
    def recall_ceiling(self) -> float:
        """Best Recall@k reachable for this query.

        With more judged relevant products than positions, perfect retrieval
        still cannot reach 1.0. Reporting the ceiling keeps a low recall from
        being read as a failure of the system.
        """
        if self.relevant_total == 0:
            return 0.0
        return min(self.k, self.relevant_total) / self.relevant_total

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "query_id": self.query_id,
            f"ndcg_at_{self.k}": self.ndcg,
            f"recall_at_{self.k}": self.recall,
            f"mrr_at_{self.k}": self.mrr,
            "relevant_total": self.relevant_total,
            "relevant_retrieved": self.relevant_retrieved,
            "recall_ceiling": self.recall_ceiling,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Macro-averaged metrics over a whole workload."""

    k: int
    relevance_threshold: float
    per_query: tuple[QueryMetrics, ...]
    mean_ndcg: float
    mean_recall: float
    mean_mrr: float

    @property
    def mean_recall_ceiling(self) -> float:
        """Macro ceiling, for reading ``mean_recall`` in context."""
        return fmean(item.recall_ceiling for item in self.per_query)

    def summary(self) -> dict[str, float]:
        """Return the shape written into metricas_desarrollo.json."""
        return {
            f"ndcg_at_{self.k}": self.mean_ndcg,
            f"recall_at_{self.k}": self.mean_recall,
            f"mrr_at_{self.k}": self.mean_mrr,
            "relevance_threshold": self.relevance_threshold,
            "recall_ceiling": self.mean_recall_ceiling,
        }

    def per_query_rows(self) -> list[dict[str, str | int | float]]:
        return [item.as_dict() for item in self.per_query]


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise MetricError(f"k debe ser un entero positivo, recibido {k!r}")


def _validate_threshold(relevance_threshold: float) -> None:
    if not math.isfinite(relevance_threshold) or relevance_threshold <= 0:
        raise MetricError("El umbral de relevancia debe ser finito y positivo")


def _validated_qrels(qrels: GradedQrels) -> dict[str, float]:
    validated: dict[str, float] = {}
    for product_id, raw in qrels.items():
        if not isinstance(product_id, str) or not product_id:
            raise MetricError("Los IDs de los juicios deben ser cadenas no vacías")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise MetricError(f"La relevancia de {product_id!r} debe ser numérica")
        relevance = float(raw)
        if not math.isfinite(relevance) or relevance < 0:
            raise MetricError(
                f"La relevancia de {product_id!r} debe ser finita y no negativa"
            )
        validated[product_id] = relevance
    return validated


def _top_k(ranking: Sequence[str], k: int) -> tuple[str, ...]:
    """Return the first ``k`` IDs, rejecting duplicates.

    A repeated product inside one ranking is always a bug —and the delivery
    checklist forbids it explicitly— so it fails here rather than quietly
    inflating a metric.
    """
    _validate_k(k)
    ids = tuple(ranking[:k])
    for product_id in ids:
        if not isinstance(product_id, str) or not product_id:
            raise MetricError("Cada resultado debe ser un product_id no vacío")
    if len(set(ids)) != len(ids):
        raise MetricError("El ranking contiene product_id duplicados dentro del top-k")
    return ids


def _discounted_cumulative_gain(relevances: Sequence[float]) -> float:
    """Standard exponential-gain DCG: an Exact is worth more than two Substitutes."""
    return sum(
        (2.0**relevance - 1.0) / math.log2(rank + 1.0)
        for rank, relevance in enumerate(relevances, start=1)
    )


def ndcg_at_k(ranking: Sequence[str], qrels: GradedQrels, *, k: int) -> float:
    """Return nDCG@k using the full graded scale."""
    judgments = _validated_qrels(qrels)
    observed = [judgments.get(product_id, 0.0) for product_id in _top_k(ranking, k)]
    ideal = sorted(judgments.values(), reverse=True)[:k]
    ideal_gain = _discounted_cumulative_gain(ideal)
    if ideal_gain == 0:
        return 0.0
    return _discounted_cumulative_gain(observed) / ideal_gain


def recall_at_k(
    ranking: Sequence[str],
    qrels: GradedQrels,
    *,
    k: int,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """Return the fraction of judged relevant products retrieved by ``k``."""
    _validate_threshold(relevance_threshold)
    judgments = _validated_qrels(qrels)
    relevant = {
        product_id
        for product_id, relevance in judgments.items()
        if relevance >= relevance_threshold
    }
    if not relevant:
        return 0.0
    return len(relevant.intersection(_top_k(ranking, k))) / len(relevant)


def mrr_at_k(
    ranking: Sequence[str],
    qrels: GradedQrels,
    *,
    k: int,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """Return the reciprocal rank of the first relevant result within ``k``."""
    _validate_threshold(relevance_threshold)
    judgments = _validated_qrels(qrels)
    for rank, product_id in enumerate(_top_k(ranking, k), start=1):
        if judgments.get(product_id, 0.0) >= relevance_threshold:
            return 1.0 / rank
    return 0.0


def recall_ceiling_at_k(
    qrels: GradedQrels,
    *,
    k: int,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> float:
    """Return the best Recall@k a perfect system could reach for these judgments.

    Not a quality metric: context for reading one. With ~25 relevant products
    per query and ten positions, Recall@10 cannot exceed ~0.40 however good the
    retrieval is.
    """
    _validate_k(k)
    _validate_threshold(relevance_threshold)
    judgments = _validated_qrels(qrels)
    relevant = sum(
        1 for relevance in judgments.values() if relevance >= relevance_threshold
    )
    if relevant == 0:
        return 0.0
    return min(k, relevant) / relevant


def evaluate_query(
    query_id: str,
    ranking: Sequence[str],
    qrels: GradedQrels,
    *,
    k: int = 10,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> QueryMetrics:
    """Compute every required metric for one query."""
    if not isinstance(query_id, str) or not query_id:
        raise MetricError("query_id debe ser una cadena no vacía")
    judgments = _validated_qrels(qrels)
    retrieved = _top_k(ranking, k)
    relevant = {
        product_id
        for product_id, relevance in judgments.items()
        if relevance >= relevance_threshold
    }
    return QueryMetrics(
        query_id=query_id,
        k=k,
        ndcg=ndcg_at_k(ranking, qrels, k=k),
        recall=recall_at_k(
            ranking, qrels, k=k, relevance_threshold=relevance_threshold
        ),
        mrr=mrr_at_k(ranking, qrels, k=k, relevance_threshold=relevance_threshold),
        relevant_total=len(relevant),
        relevant_retrieved=len(relevant.intersection(retrieved)),
        retrieved=len(retrieved),
    )


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels_by_query: Mapping[str, GradedQrels],
    *,
    k: int = 10,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> EvaluationReport:
    """Evaluate every judged query and macro-average each metric.

    A query with judgments but no ranking scores zero: an incomplete run must
    not look stronger than it is. A ranking for an unknown query is an error,
    because it almost always means the identifiers were joined wrongly —the
    exact failure ``data.py`` exists to prevent.
    """
    _validate_k(k)
    _validate_threshold(relevance_threshold)
    if not qrels_by_query:
        raise MetricError("No hay juicios de relevancia que evaluar")
    unknown = set(rankings) - set(qrels_by_query)
    if unknown:
        raise MetricError(
            f"Hay rankings sin juicios para: {', '.join(sorted(unknown))}. "
            "Suele indicar un cruce incorrecto de identificadores."
        )

    per_query = tuple(
        evaluate_query(
            query_id,
            rankings.get(query_id, ()),
            qrels,
            k=k,
            relevance_threshold=relevance_threshold,
        )
        for query_id, qrels in sorted(qrels_by_query.items())
    )
    return EvaluationReport(
        k=k,
        relevance_threshold=relevance_threshold,
        per_query=per_query,
        mean_ndcg=fmean(item.ndcg for item in per_query),
        mean_recall=fmean(item.recall for item in per_query),
        mean_mrr=fmean(item.mrr for item in per_query),
    )
