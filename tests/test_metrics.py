"""Ranking metrics checked against values computed by hand (RF-19).

An implementation that only agrees with itself proves nothing, so the reference
numbers here are worked out step by step in the docstrings.
"""

from __future__ import annotations

import math

import pytest

from aurum_market.evaluation.metrics import (
    MetricError,
    evaluate_query,
    evaluate_rankings,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    recall_ceiling_at_k,
)

# Un caso pequeño con las cuatro etiquetas ESCI representadas.
QRELS = {"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0}


class TestNdcg:
    def test_perfect_ranking_scores_one(self) -> None:
        assert ndcg_at_k(["A", "B", "C"], QRELS, k=3) == pytest.approx(1.0)

    def test_hand_computed_value(self) -> None:
        """ranking = [A(3), D(0), B(2)] con k=3.

        DCG  = (2^3-1)/log2(2) + (2^0-1)/log2(3) + (2^2-1)/log2(4)
             = 7/1 + 0/1.584963 + 3/2
             = 8.5

        IDCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
             = 7/1 + 3/1.584963 + 1/2
             = 9.392789

        nDCG = 8.5 / 9.392789 = 0.904949
        """
        dcg = 7 / math.log2(2) + 0 / math.log2(3) + 3 / math.log2(4)
        idcg = 7 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
        assert dcg == pytest.approx(8.5)
        assert idcg == pytest.approx(9.392789260714372)
        assert ndcg_at_k(["A", "D", "B"], QRELS, k=3) == pytest.approx(0.9049, abs=1e-4)

    def test_position_matters(self) -> None:
        """El mismo conjunto de resultados vale menos peor ordenado."""
        good = ndcg_at_k(["A", "B"], QRELS, k=2)
        bad = ndcg_at_k(["B", "A"], QRELS, k=2)
        assert good > bad

    def test_an_exact_outweighs_two_substitutes(self) -> None:
        """La ganancia exponencial 2^rel-1 hace que 3 valga 7 y 2 valga 3."""
        qrels = {"exact": 3.0, "sub1": 2.0, "sub2": 2.0}
        assert ndcg_at_k(["exact"], qrels, k=1) > ndcg_at_k(["sub1"], qrels, k=1)

    def test_a_complement_still_contributes_gain(self) -> None:
        """nDCG usa toda la escala: un Complement no es lo mismo que un Irrelevant."""
        assert ndcg_at_k(["C"], QRELS, k=1) > ndcg_at_k(["D"], QRELS, k=1)
        assert ndcg_at_k(["D"], QRELS, k=1) == 0.0

    def test_unjudged_products_count_as_zero(self) -> None:
        assert ndcg_at_k(["desconocido"], QRELS, k=1) == 0.0

    def test_qrels_without_any_gain_yield_zero(self) -> None:
        assert ndcg_at_k(["X"], {"X": 0.0}, k=1) == 0.0


class TestRecall:
    def test_threshold_two_ignores_complements(self) -> None:
        """ADR-004: relevante = Exact o Substitute. Relevantes = {A, B}."""
        assert recall_at_k(["A", "B"], QRELS, k=2) == pytest.approx(1.0)
        assert recall_at_k(["C", "D"], QRELS, k=2) == pytest.approx(0.0)

    def test_hand_computed_value(self) -> None:
        """ranking = [D, C, B]: de {A, B} se recupera solo B → 1/2."""
        assert recall_at_k(["D", "C", "B"], QRELS, k=3) == pytest.approx(0.5)

    def test_lowering_the_threshold_would_inflate_recall(self) -> None:
        """Justo lo que ADR-004 evita: contar accesorios como aciertos."""
        strict = recall_at_k(["C"], QRELS, k=1, relevance_threshold=2.0)
        loose = recall_at_k(["C"], QRELS, k=1, relevance_threshold=1.0)
        assert strict == 0.0
        assert loose > strict

    def test_no_relevant_judgments_yield_zero(self) -> None:
        assert recall_at_k(["X"], {"X": 1.0}, k=1) == 0.0

    def test_results_beyond_k_do_not_count(self) -> None:
        assert recall_at_k(["D", "C", "A", "B"], QRELS, k=2) == pytest.approx(0.0)


class TestMrr:
    def test_first_position_scores_one(self) -> None:
        assert mrr_at_k(["A", "B"], QRELS, k=2) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("ranking", "expected"),
        [
            (["D", "A"], 1 / 2),
            (["D", "C", "A"], 1 / 3),
            (["D", "C", "D2", "B"], 1 / 4),
        ],
    )
    def test_reciprocal_of_the_first_relevant_position(
        self, ranking: list[str], expected: float
    ) -> None:
        assert mrr_at_k(ranking, QRELS, k=4) == pytest.approx(expected)

    def test_no_relevant_result_scores_zero(self) -> None:
        assert mrr_at_k(["C", "D"], QRELS, k=2) == 0.0

    def test_only_the_first_relevant_matters(self) -> None:
        assert mrr_at_k(["A", "B"], QRELS, k=2) == mrr_at_k(["A", "D"], QRELS, k=2)


class TestRecallCeiling:
    def test_ceiling_is_one_when_positions_suffice(self) -> None:
        assert recall_ceiling_at_k(QRELS, k=10) == pytest.approx(1.0)

    def test_ceiling_drops_when_relevants_exceed_positions(self) -> None:
        """25 relevantes y 10 posiciones: el máximo alcanzable es 0,40."""
        qrels = {f"P{i}": 3.0 for i in range(25)}
        assert recall_ceiling_at_k(qrels, k=10) == pytest.approx(0.4)

    def test_ceiling_ignores_products_below_the_threshold(self) -> None:
        qrels = {"A": 3.0, **{f"C{i}": 1.0 for i in range(50)}}
        assert recall_ceiling_at_k(qrels, k=10) == pytest.approx(1.0)


class TestValidation:
    def test_duplicate_products_in_a_ranking_are_rejected(self) -> None:
        """Punto 4 de "Antes de entregar": diez IDs únicos."""
        with pytest.raises(MetricError, match="duplicados"):
            ndcg_at_k(["A", "A"], QRELS, k=2)

    @pytest.mark.parametrize("k", [0, -1, True])
    def test_k_must_be_a_positive_integer(self, k: object) -> None:
        with pytest.raises(MetricError, match="entero positivo"):
            ndcg_at_k(["A"], QRELS, k=k)  # type: ignore[arg-type]

    def test_negative_relevance_is_rejected(self) -> None:
        with pytest.raises(MetricError, match="no negativa"):
            ndcg_at_k(["A"], {"A": -1.0}, k=1)

    def test_an_empty_product_id_is_rejected(self) -> None:
        with pytest.raises(MetricError, match="product_id no vacío"):
            ndcg_at_k([""], QRELS, k=1)


class TestWorkloadEvaluation:
    def test_macro_averages_across_queries(self) -> None:
        report = evaluate_rankings(
            {"Q1": ["A", "B"], "Q2": ["C", "D"]},
            {"Q1": QRELS, "Q2": QRELS},
            k=2,
        )
        assert report.mean_recall == pytest.approx(0.5)  # (1.0 + 0.0) / 2
        assert report.mean_mrr == pytest.approx(0.5)  # (1.0 + 0.0) / 2
        assert len(report.per_query) == 2

    def test_a_missing_ranking_scores_zero_instead_of_being_skipped(self) -> None:
        """Una ejecución incompleta no puede parecer más fuerte de lo que es."""
        report = evaluate_rankings({"Q1": ["A"]}, {"Q1": QRELS, "Q2": QRELS}, k=1)
        assert len(report.per_query) == 2
        assert report.per_query[1].ndcg == 0.0

    def test_a_ranking_for_an_unknown_query_is_an_error(self) -> None:
        """Casi siempre significa que se cruzaron mal los identificadores."""
        with pytest.raises(MetricError, match="identificadores"):
            evaluate_rankings({"DEV-13357": ["A"]}, {"13357": QRELS}, k=1)

    def test_summary_matches_the_artifact_contract(self) -> None:
        report = evaluate_rankings({"Q1": ["A", "B"]}, {"Q1": QRELS}, k=10)
        summary = report.summary()
        assert "ndcg_at_10" in summary
        assert "recall_at_10" in summary
        assert "mrr_at_10" in summary
        assert summary["relevance_threshold"] == 2.0

    def test_per_query_metrics_expose_the_counts_behind_them(self) -> None:
        metrics = evaluate_query("Q1", ["A", "D"], QRELS, k=2)
        assert metrics.relevant_total == 2  # A y B
        assert metrics.relevant_retrieved == 1  # solo A
        assert metrics.recall == pytest.approx(0.5)
