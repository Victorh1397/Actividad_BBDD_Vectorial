"""ANN fidelity and brand-filter verification (RF-14, RF-20, RF-22).

Two measurements that only make sense together with the ranking metrics:

* **Fidelity** compares the IDs Qdrant returns against the exact oracle, which
  is what separates *"the index dropped a good candidate"* from *"the model
  never understood the query"*. Both run through the same retriever and differ
  only in the store, so nothing else can explain a discrepancy.
* **Filter compliance** checks that a brand-restricted search returns that
  brand and nothing else — point 2 of the delivery checklist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean

from ..contracts import RetrievalQuery
from ..search import DenseRetriever


@dataclass(frozen=True, slots=True)
class QueryFidelity:
    """Overlap between the engine's ranking and the exact one, for one query."""

    query_id: str
    k: int
    engine_ids: tuple[str, ...]
    exact_ids: tuple[str, ...]

    @property
    def overlap(self) -> tuple[str, ...]:
        exact = set(self.exact_ids)
        return tuple(pid for pid in self.engine_ids if pid in exact)

    @property
    def recall(self) -> float:
        """Fraction of the exact top-k the engine also returned."""
        if not self.exact_ids:
            return 0.0
        return len(self.overlap) / min(self.k, len(self.exact_ids))

    @property
    def missed(self) -> tuple[str, ...]:
        """What the exact search found and the index did not.

        These are the candidates lost to approximation, and the raw material
        for attributing a failure to the index rather than to the model.
        """
        engine = set(self.engine_ids)
        return tuple(pid for pid in self.exact_ids if pid not in engine)

    @property
    def rank_agreement(self) -> float:
        """Fraction of positions holding the very same product.

        Recall can be 1.0 while the order differs; this catches that.
        """
        if not self.exact_ids:
            return 0.0
        pairs = zip(self.engine_ids, self.exact_ids, strict=False)
        return sum(1 for a, b in pairs if a == b) / min(
            len(self.engine_ids), len(self.exact_ids)
        )


@dataclass(frozen=True, slots=True)
class FidelityReport:
    """Fidelity across a workload, at one ``ef_search`` setting."""

    k: int
    ef_search: int | None
    per_query: tuple[QueryFidelity, ...]

    @property
    def mean_recall(self) -> float:
        return fmean(item.recall for item in self.per_query) if self.per_query else 0.0

    @property
    def mean_rank_agreement(self) -> float:
        if not self.per_query:
            return 0.0
        return fmean(item.rank_agreement for item in self.per_query)

    @property
    def perfect_queries(self) -> int:
        return sum(1 for item in self.per_query if item.recall >= 1.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "ef_search": self.ef_search,
            "ann_fidelity": self.mean_recall,
            "rank_agreement": self.mean_rank_agreement,
            "perfect_queries": self.perfect_queries,
            "queries": len(self.per_query),
            "per_query": [
                {
                    "query_id": item.query_id,
                    "fidelity": item.recall,
                    "missed": list(item.missed),
                }
                for item in self.per_query
            ],
        }


def measure_fidelity(
    engine: DenseRetriever,
    oracle: DenseRetriever,
    queries: Sequence[RetrievalQuery],
    *,
    k: int = 10,
    ef_search: int | None = None,
) -> FidelityReport:
    """Compare engine and oracle rankings over the same queries.

    Both retrievers encode with the same model, so any difference comes from
    the index and nothing else.
    """
    results = []
    for query in queries:
        engine_hits = _search(engine, query.text, top_k=k, ef_search=ef_search)
        exact_hits = oracle.search(query.text, top_k=k)
        results.append(
            QueryFidelity(
                query_id=query.query_id,
                k=k,
                engine_ids=tuple(hit.product_id for hit in engine_hits),
                exact_ids=tuple(hit.product_id for hit in exact_hits),
            )
        )
    return FidelityReport(k=k, ef_search=ef_search, per_query=tuple(results))


def _search(retriever: DenseRetriever, text: str, *, top_k: int, ef_search: int | None):
    """Search, passing ``ef_search`` only to stores that understand it."""
    if ef_search is None:
        return retriever.search(text, top_k=top_k)
    store = retriever._store
    matrix = retriever._encoder.encode([text], role="query")
    return store.search_vector(matrix.vectors[0], top_k=top_k, ef_search=ef_search)


def sweep_ef_search(
    engine: DenseRetriever,
    oracle: DenseRetriever,
    queries: Sequence[RetrievalQuery],
    *,
    values: Sequence[int] = (16, 32, 64, 128, 256),
    k: int = 10,
) -> list[FidelityReport]:
    """Trace the fidelity curve across ``ef_search`` values.

    ``ef_search`` is the only HNSW parameter adjustable without rebuilding the
    graph, which makes it the one worth sweeping: it is the knob a production
    system would actually turn (RF-08).
    """
    return [
        measure_fidelity(engine, oracle, queries, k=k, ef_search=value)
        for value in values
    ]


@dataclass(frozen=True, slots=True)
class FilterCheck:
    """Whether one brand-restricted query honoured its constraint."""

    query_id: str
    brand: str
    returned: int
    matching: int
    offending_brands: tuple[str, ...] = field(default_factory=tuple)

    @property
    def compliant(self) -> bool:
        """Every result belongs to the requested brand.

        An empty result set counts as compliant: the filter did its job, the
        catalog simply had nothing (RF-15).
        """
        return self.matching == self.returned

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "brand": self.brand,
            "returned": self.returned,
            "matching": self.matching,
            "compliant": self.compliant,
            "offending_brands": list(self.offending_brands),
        }


def check_brand_filters(
    retriever: DenseRetriever,
    queries: Sequence[RetrievalQuery],
    *,
    k: int = 10,
) -> list[FilterCheck]:
    """Run the filtered workload and verify no other brand leaks through."""
    checks = []
    for query in queries:
        if not query.brand:
            raise ValueError(f"{query.query_id} no declara marca que filtrar")
        hits = retriever.search(query.text, top_k=k, brand=query.brand)
        offending = tuple(
            sorted({hit.brand for hit in hits if hit.brand != query.brand})
        )
        checks.append(
            FilterCheck(
                query_id=query.query_id,
                brand=query.brand,
                returned=len(hits),
                matching=sum(1 for hit in hits if hit.brand == query.brand),
                offending_brands=offending,
            )
        )
    return checks


def summarize_filters(checks: Sequence[FilterCheck]) -> Mapping[str, object]:
    """Aggregate filter compliance for the metrics artifact."""
    return {
        "queries": len(checks),
        "compliant": sum(1 for check in checks if check.compliant),
        "all_compliant": all(check.compliant for check in checks),
        "detail": [check.as_dict() for check in checks],
    }
