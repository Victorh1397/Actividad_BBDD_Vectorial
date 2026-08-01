"""ANN fidelity, brand-filter compliance and latency (RF-14, RF-20, RF-21)."""

from __future__ import annotations

from typing import ClassVar

import pytest

from aurum_market.contracts import RetrievalQuery, SearchHit
from aurum_market.evaluation.fidelity import (
    FidelityReport,
    FilterCheck,
    QueryFidelity,
    check_brand_filters,
    summarize_filters,
)
from aurum_market.evaluation.latency import (
    describe_environment,
    measure_latency,
)


def fidelity(engine: list[str], exact: list[str], k: int = 5) -> QueryFidelity:
    return QueryFidelity(
        query_id="Q1", k=k, engine_ids=tuple(engine), exact_ids=tuple(exact)
    )


class TestQueryFidelity:
    def test_identical_rankings_score_one(self) -> None:
        item = fidelity(["A", "B", "C"], ["A", "B", "C"])
        assert item.recall == 1.0
        assert item.rank_agreement == 1.0
        assert item.missed == ()

    def test_missing_one_candidate(self) -> None:
        """Lo que el oráculo encuentra y el índice no: pérdida por aproximación."""
        item = fidelity(["A", "B", "D"], ["A", "B", "C"])
        assert item.recall == pytest.approx(2 / 3)
        assert item.missed == ("C",)

    def test_same_set_in_different_order(self) -> None:
        """El recall no ve el orden; rank_agreement sí."""
        item = fidelity(["C", "B", "A"], ["A", "B", "C"])
        assert item.recall == 1.0
        assert item.rank_agreement == pytest.approx(1 / 3)

    def test_nothing_in_common(self) -> None:
        item = fidelity(["X", "Y"], ["A", "B"])
        assert item.recall == 0.0
        assert set(item.missed) == {"A", "B"}

    def test_an_empty_oracle_scores_zero_instead_of_dividing_by_zero(self) -> None:
        assert fidelity(["A"], []).recall == 0.0


class TestFidelityReport:
    def test_aggregates_across_queries(self) -> None:
        report = FidelityReport(
            k=3,
            ef_search=128,
            per_query=(
                fidelity(["A", "B", "C"], ["A", "B", "C"], k=3),
                fidelity(["A", "B", "X"], ["A", "B", "C"], k=3),
            ),
        )
        assert report.mean_recall == pytest.approx((1.0 + 2 / 3) / 2)
        assert report.perfect_queries == 1

    def test_serializes_the_missed_candidates(self) -> None:
        """Sin los IDs perdidos no se puede atribuir un fallo al índice (RF-24)."""
        report = FidelityReport(
            k=3, ef_search=64, per_query=(fidelity(["A"], ["A", "B"], k=3),)
        )
        payload = report.as_dict()
        assert payload["ef_search"] == 64
        assert payload["per_query"][0]["missed"] == ["B"]

    def test_an_empty_report_does_not_explode(self) -> None:
        report = FidelityReport(k=10, ef_search=None, per_query=())
        assert report.mean_recall == 0.0
        assert report.perfect_queries == 0


class FakeRetriever:
    """Returns a fixed catalog, honouring the brand filter."""

    def __init__(self, products: list[tuple[str, str]]) -> None:
        self._products = products

    def search(
        self, query_text: str, *, top_k: int = 10, brand: str | None = None
    ) -> list[SearchHit]:
        chosen = [(pid, b) for pid, b in self._products if brand is None or b == brand][
            :top_k
        ]
        return [
            SearchHit(
                rank=rank,
                record_id=f"{i:08x}-a995-56d0-ba03-559885ccef39",
                product_id=pid,
                title=f"Producto {pid}",
                brand=b,
                native_score=1.0 - rank * 0.01,
                score_kind="similarity",
                higher_is_better=True,
            )
            for rank, (i, (pid, b)) in enumerate(enumerate(chosen), start=1)
        ]


class LeakyRetriever(FakeRetriever):
    """Ignores the brand filter, as a post-filter mistake would."""

    def search(self, query_text: str, *, top_k: int = 10, brand: str | None = None):
        return super().search(query_text, top_k=top_k, brand=None)


class TestBrandFilters:
    CATALOG: ClassVar[list[tuple[str, str]]] = [
        ("P1", "Einhell"),
        ("P2", "Bosch"),
        ("P3", "Einhell"),
    ]

    def query(self, brand: str = "Einhell") -> RetrievalQuery:
        return RetrievalQuery(
            query_id="FILTER-001", text="taladro", brand=brand, query_type="filtered"
        )

    def test_a_compliant_filter_returns_only_that_brand(self) -> None:
        checks = check_brand_filters(FakeRetriever(self.CATALOG), [self.query()])
        assert checks[0].compliant
        assert checks[0].offending_brands == ()

    def test_a_leaking_filter_is_caught(self) -> None:
        """Punto 2 de "Antes de entregar": nunca otra marca."""
        checks = check_brand_filters(LeakyRetriever(self.CATALOG), [self.query()])
        assert not checks[0].compliant
        assert "Bosch" in checks[0].offending_brands

    def test_an_absent_brand_yields_an_empty_but_compliant_result(self) -> None:
        """El filtro cumplió; el catálogo simplemente no tenía nada (RF-15)."""
        checks = check_brand_filters(
            FakeRetriever(self.CATALOG), [self.query(brand="Makita")]
        )
        assert checks[0].returned == 0
        assert checks[0].compliant

    def test_a_query_without_a_brand_is_rejected(self) -> None:
        query = RetrievalQuery(query_id="FILTER-001", text="taladro")
        with pytest.raises(ValueError, match="no declara marca"):
            check_brand_filters(FakeRetriever(self.CATALOG), [query])

    def test_the_summary_reports_overall_compliance(self) -> None:
        checks = [
            FilterCheck("F1", "Einhell", returned=10, matching=10),
            FilterCheck(
                "F2", "Apple", returned=10, matching=9, offending_brands=("HP",)
            ),
        ]
        summary = summarize_filters(checks)
        assert summary["queries"] == 2
        assert summary["compliant"] == 1
        assert summary["all_compliant"] is False


class TestLatency:
    def test_reports_percentiles_over_the_declared_repetitions(self) -> None:
        summary = measure_latency(
            lambda value: value * 2,
            [1, 2, 3],
            label="doblar",
            warmup_repetitions=1,
            repetitions=4,
        )
        assert summary.count == 12  # 3 entradas x 4 repeticiones
        assert summary.p50_ms <= summary.p95_ms
        assert summary.min_ms <= summary.p50_ms <= summary.max_ms

    def test_warm_up_runs_are_excluded_from_the_samples(self) -> None:
        """Incluirlas reportaría un coste de arranque como si fuera estable."""
        calls: list[int] = []
        measure_latency(
            calls.append, [1, 2], label="x", warmup_repetitions=3, repetitions=2
        )
        assert len(calls) == 3 * 2 + 2 * 2  # calentamiento + medición
        summary = measure_latency(
            lambda v: v, [1, 2], label="x", warmup_repetitions=3, repetitions=2
        )
        assert summary.count == 4  # solo las medidas

    def test_serializes_with_its_conditions(self) -> None:
        """Una latencia sin sus condiciones no es reproducible (RF-21)."""
        payload = measure_latency(
            lambda v: v, [1], label="x", warmup_repetitions=2, repetitions=3
        ).as_dict()
        assert payload["warmup_repetitions"] == 2
        assert payload["repetitions"] == 3
        assert "p50_ms" in payload and "p95_ms" in payload

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"repetitions": 0}, "repetitions"),
            ({"warmup_repetitions": -1}, "warmup"),
        ],
    )
    def test_invalid_settings_are_rejected(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            measure_latency(lambda v: v, [1], label="x", **kwargs)

    def test_no_inputs_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="al menos una entrada"):
            measure_latency(lambda v: v, [], label="x")

    def test_environment_records_what_the_numbers_depend_on(self) -> None:
        environment = describe_environment(profile="full")
        assert environment["python_version"]
        assert environment["platform"]
        assert environment["profile"] == "full"
