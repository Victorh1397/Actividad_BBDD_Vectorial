"""Measurement layer: ranking quality, ANN fidelity, latency and error attribution."""

from __future__ import annotations

from .fidelity import (
    FidelityReport,
    FilterCheck,
    QueryFidelity,
    check_brand_filters,
    measure_fidelity,
    summarize_filters,
    sweep_ef_search,
)
from .latency import LatencySummary, describe_environment, measure_latency
from .metrics import (
    EvaluationReport,
    QueryMetrics,
    evaluate_query,
    evaluate_rankings,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    recall_ceiling_at_k,
)

__all__ = [
    "EvaluationReport",
    "FidelityReport",
    "FilterCheck",
    "LatencySummary",
    "QueryFidelity",
    "QueryMetrics",
    "check_brand_filters",
    "describe_environment",
    "evaluate_query",
    "evaluate_rankings",
    "measure_fidelity",
    "measure_latency",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "recall_ceiling_at_k",
    "summarize_filters",
    "sweep_ef_search",
]
