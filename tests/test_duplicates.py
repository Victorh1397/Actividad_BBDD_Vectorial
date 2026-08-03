"""Duplicate detection rule and calibration (RF-17, RF-23)."""

from __future__ import annotations

import pytest

from aurum_market.contracts import ContractError, IncomingListing, SearchHit
from aurum_market.duplicates import (
    DuplicateError,
    ListingEvidence,
    calibrate,
    compose_listing,
    decide,
    error_analysis,
    gather_evidence,
    predict,
    score_threshold,
)


def make_listing(
    incoming_id: str = "DEV-DUP-001",
    *,
    is_duplicate: bool | None = True,
    reference: str | None = "B000G3T55M",
    brand: str = "NIKE",
    color: str = "Negro",
) -> IncomingListing:
    return IncomingListing(
        incoming_id=incoming_id,
        title="NIKE Legasee Legging Swoosh",
        brand=brand,
        color=color,
        text="NIKE Legasee Legging Swoosh. Marca: . Color: Negro",
        is_duplicate=is_duplicate,
        reference_product_id=reference if is_duplicate else None,
    )


def make_hit(product_id: str, score: float, rank: int = 1) -> SearchHit:
    return SearchHit(
        rank=rank,
        record_id="000bd6e8-a995-56d0-ba03-559885ccef39",
        product_id=product_id,
        title=f"Producto {product_id}",
        brand="NIKE",
        native_score=score,
        score_kind="similarity",
        higher_is_better=True,
    )


def make_evidence(
    score: float,
    *,
    runner_up: float | None = None,
    is_duplicate: bool | None = True,
    product_id: str = "B000G3T55M",
    incoming_id: str = "DEV-DUP-001",
) -> ListingEvidence:
    hits = [make_hit(product_id, score)]
    if runner_up is not None:
        hits.append(make_hit("B0OTHER", runner_up, rank=2))
    return ListingEvidence(
        listing=make_listing(
            incoming_id,
            is_duplicate=is_duplicate,
            reference=product_id if is_duplicate else None,
        ),
        candidates=tuple(hits),
    )


class TestListingComposition:
    def test_mirrors_the_catalog_format(self) -> None:
        """El catálogo se indexó así; la ficha entrante debe componerse igual."""
        composed = compose_listing(make_listing(), "title_brand_color")
        assert composed == "NIKE Legasee Legging Swoosh. Marca: NIKE. Color: Negro"

    def test_absent_metadata_is_skipped(self) -> None:
        composed = compose_listing(
            make_listing(brand="", color=""), "title_brand_color"
        )
        assert composed == "NIKE Legasee Legging Swoosh"

    def test_title_only_uses_just_the_title(self) -> None:
        assert compose_listing(make_listing(), "title_only") == (
            "NIKE Legasee Legging Swoosh"
        )

    def test_raw_text_falls_back_to_the_title(self) -> None:
        listing = IncomingListing(
            incoming_id="X", title="Un título", brand="", color="", text=""
        )
        assert compose_listing(listing, "raw_text") == "Un título"

    def test_an_unknown_strategy_is_rejected(self) -> None:
        with pytest.raises(DuplicateError, match="desconocida"):
            compose_listing(make_listing(), "titulo_y_precio")  # type: ignore[arg-type]


class TestDecisionRule:
    def test_above_the_threshold_is_a_duplicate(self) -> None:
        decision = decide(make_evidence(0.95), threshold=0.9191)
        assert decision.predicted_duplicate
        assert decision.matched_product_id == "B000G3T55M"

    def test_below_the_threshold_is_new(self) -> None:
        decision = decide(make_evidence(0.88), threshold=0.9191)
        assert not decision.predicted_duplicate
        assert decision.matched_product_id == ""

    def test_exactly_at_the_threshold_counts_as_duplicate(self) -> None:
        """El criterio es >=; dejarlo ambiguo haría la regla irreproducible."""
        assert decide(make_evidence(0.9191), threshold=0.9191).predicted_duplicate

    def test_positive_prediction_names_a_candidate(self) -> None:
        """Punto 5 de "Antes de entregar"."""
        decision = decide(make_evidence(0.96), threshold=0.9191)
        assert decision.predicted_duplicate
        assert decision.matched_product_id

    def test_a_positive_without_candidate_cannot_be_constructed(self) -> None:
        """El tipo lo impide, así que la regla no puede violarlo por error."""
        evidence = ListingEvidence(listing=make_listing(), candidates=())
        decision = decide(evidence, threshold=0.9191)
        assert not decision.predicted_duplicate
        with pytest.raises(ContractError):
            from aurum_market.contracts import DuplicateDecision

            DuplicateDecision("X", True, "", 0.99)

    def test_the_rule_is_deterministic(self) -> None:
        evidence = make_evidence(0.95, runner_up=0.80)
        first = decide(evidence, threshold=0.9191)
        second = decide(evidence, threshold=0.9191)
        assert first == second

    def test_the_margin_travels_with_the_decision(self) -> None:
        """Sirve para reportar cuán ajustada fue, aunque no decida."""
        decision = decide(make_evidence(0.95, runner_up=0.90), threshold=0.9191)
        assert decision.margin == pytest.approx(0.05)


class TestThresholdScoring:
    def evidences(self) -> list[ListingEvidence]:
        return [
            make_evidence(0.96, is_duplicate=True, incoming_id="D1"),
            make_evidence(0.95, is_duplicate=True, incoming_id="D2"),
            make_evidence(0.86, is_duplicate=False, incoming_id="N1"),
            make_evidence(0.88, is_duplicate=False, incoming_id="N2"),
        ]

    def test_a_perfect_threshold_scores_one(self) -> None:
        outcome = score_threshold(self.evidences(), 0.91)
        assert outcome.precision == 1.0
        assert outcome.recall == 1.0
        assert outcome.f1 == 1.0
        assert (outcome.true_positives, outcome.false_negatives) == (2, 0)

    def test_a_low_threshold_produces_false_positives(self) -> None:
        outcome = score_threshold(self.evidences(), 0.50)
        assert outcome.false_positives == 2
        assert outcome.precision == pytest.approx(0.5)
        assert outcome.recall == 1.0

    def test_a_high_threshold_produces_false_negatives(self) -> None:
        outcome = score_threshold(self.evidences(), 0.99)
        assert outcome.false_negatives == 2
        assert outcome.recall == 0.0

    def test_a_positive_pointing_at_the_wrong_product_is_flagged(self) -> None:
        """Detecta que es duplicado pero señala otro: cuenta y se reporta."""
        evidence = ListingEvidence(
            listing=make_listing("D1", is_duplicate=True, reference="B000G3T55M"),
            candidates=(make_hit("B0OTRO", 0.96),),
        )
        outcome = score_threshold([evidence], 0.91)
        assert outcome.true_positives == 1
        assert outcome.wrong_candidate == 1

    def test_unlabelled_listings_cannot_be_scored(self) -> None:
        evidence = make_evidence(0.95, is_duplicate=None)
        with pytest.raises(DuplicateError, match="no está etiquetado"):
            score_threshold([evidence], 0.91)


class TestCalibration:
    def separable(self) -> list[ListingEvidence]:
        return [
            make_evidence(s, is_duplicate=True, incoming_id=f"D{i}")
            for i, s in enumerate([0.9484, 0.9532, 0.9662])
        ] + [
            make_evidence(s, is_duplicate=False, incoming_id=f"N{i}")
            for i, s in enumerate([0.8521, 0.8712, 0.8898])
        ]

    def test_separable_classes_yield_the_midpoint(self) -> None:
        """Con F1 perfecto en todo el hueco, la robustez elige el punto."""
        result = calibrate(self.separable())
        assert result.separation == pytest.approx(0.9484 - 0.8898)
        assert result.threshold == pytest.approx((0.9484 + 0.8898) / 2)

    def test_the_chosen_threshold_classifies_development_perfectly(self) -> None:
        evidences = self.separable()
        result = calibrate(evidences)
        outcome = score_threshold(evidences, result.threshold)
        assert outcome.f1 == 1.0
        assert outcome.false_positives == 0
        assert outcome.false_negatives == 0

    def test_overlapping_classes_fall_back_to_maximising_f1(self) -> None:
        evidences = [
            make_evidence(0.95, is_duplicate=True, incoming_id="D1"),
            make_evidence(0.88, is_duplicate=True, incoming_id="D2"),
            make_evidence(0.90, is_duplicate=False, incoming_id="N1"),
            make_evidence(0.85, is_duplicate=False, incoming_id="N2"),
        ]
        result = calibrate(evidences)
        assert result.separation < 0
        assert any("se solapan" in note for note in result.notes)

    def test_the_notes_explain_the_choice(self) -> None:
        """El enunciado exige justificar el umbral, no solo declararlo."""
        result = calibrate(self.separable())
        assert result.notes
        assert "punto medio" in result.notes[0]

    def test_calibration_needs_both_classes(self) -> None:
        only_duplicates = [make_evidence(0.95, is_duplicate=True)]
        with pytest.raises(DuplicateError, match="positivos y negativos"):
            calibrate(only_duplicates)

    def test_empty_evidence_is_rejected(self) -> None:
        with pytest.raises(DuplicateError, match="No hay evidencia"):
            calibrate([])

    def test_serializes_the_sweep_and_the_distributions(self) -> None:
        payload = calibrate(self.separable()).as_dict()
        assert payload["threshold"] > 0
        assert payload["sweep"]
        assert payload["duplicate_scores"]["min"] == pytest.approx(0.9484)
        assert payload["new_scores"]["max"] == pytest.approx(0.8898)


class TestErrorAnalysis:
    def test_separates_false_positives_from_false_negatives(self) -> None:
        """RF-23: tienen costes de negocio distintos y se reportan aparte."""
        evidences = [
            make_evidence(0.95, is_duplicate=False, incoming_id="N1"),  # FP
            make_evidence(0.80, is_duplicate=True, incoming_id="D1"),  # FN
        ]
        analysis = error_analysis(evidences, threshold=0.91)
        assert len(analysis["false_positives"]) == 1
        assert len(analysis["false_negatives"]) == 1
        assert analysis["false_positives"][0]["incoming_id"] == "N1"

    def test_explains_what_each_error_costs(self) -> None:
        analysis = error_analysis([make_evidence(0.95)], threshold=0.91)
        note = analysis["cost_note"]
        assert "bloquea una publicación" in note
        assert "degrada el catálogo" in note


class FakeRetriever:
    def __init__(self, score: float) -> None:
        self.score = score
        self.queries: list[str] = []

    def search(self, query_text: str, *, top_k: int = 10, brand: str | None = None):
        self.queries.append(query_text)
        return [make_hit("B000G3T55M", self.score, rank=i + 1) for i in range(top_k)]


class TestEvidenceGathering:
    def test_candidates_come_from_the_vector_store(self) -> None:
        """El enunciado exige que la base vectorial genere los candidatos."""
        retriever = FakeRetriever(0.95)
        evidences = gather_evidence(retriever, [make_listing()])
        assert retriever.queries
        assert len(evidences[0].candidates) == 2

    def test_the_query_uses_the_catalog_format(self) -> None:
        retriever = FakeRetriever(0.95)
        gather_evidence(retriever, [make_listing()], strategy="title_brand_color")
        assert retriever.queries[0] == (
            "NIKE Legasee Legging Swoosh. Marca: NIKE. Color: Negro"
        )

    def test_predict_applies_a_frozen_threshold(self) -> None:
        retriever = FakeRetriever(0.95)
        evidences = gather_evidence(
            retriever, [make_listing("EVAL-DUP-001", is_duplicate=None, reference=None)]
        )
        decisions = predict(evidences, threshold=0.9191)
        assert decisions[0].predicted_duplicate
        assert decisions[0].incoming_id == "EVAL-DUP-001"
