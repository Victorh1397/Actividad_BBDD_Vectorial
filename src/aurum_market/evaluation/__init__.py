"""Measurement layer: ranking quality, ANN fidelity, latency and error attribution."""

from __future__ import annotations

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
    "QueryMetrics",
    "evaluate_query",
    "evaluate_rankings",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "recall_ceiling_at_k",
]
