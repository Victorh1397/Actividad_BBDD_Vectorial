"""Duplicate detection for incoming listings (RF-17, RF-23).

The vector store is always the candidate generator, as the statement requires.
The rule on top of it is deliberately simple — a threshold on the similarity of
the best candidate — because the data says nothing more elaborate is warranted:
on the development set the two classes do not overlap.

One design decision does the heavy lifting: **the incoming listing is composed
with the same strategy the catalog was indexed with**. Comparing a raw incoming
`text` against a catalog encoded as `title + brand + color` compares two
different things, and it shows — the separation between duplicates and new
products goes from 0.0032 to 0.0586 once both sides share a format.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean

from .contracts import DuplicateDecision, IncomingListing, SearchHit
from .search import Retriever
from .text import TextStrategy

# Se buscan dos candidatos, no uno: el segundo permite calcular el margen y
# reportar cuán ajustada fue la decisión, aunque el margen no forme parte de
# la regla (ver ADR-008).
CANDIDATES_PER_LISTING = 2


class DuplicateError(ValueError):
    """Raised when a listing cannot be evaluated for duplication."""


def compose_listing(listing: IncomingListing, strategy: TextStrategy) -> str:
    """Render an incoming listing the same way its catalog counterpart was.

    Mirrors ``text.compose`` field by field. Absent metadata is skipped rather
    than rendered as an empty label, exactly as in the catalog (P-07).
    """
    if strategy == "title_only":
        composed = listing.title
    elif strategy == "raw_text":
        composed = listing.text or listing.title
    elif strategy == "title_brand_color":
        parts = [listing.title]
        if listing.brand:
            parts.append(f"Marca: {listing.brand}")
        if listing.color:
            parts.append(f"Color: {listing.color}")
        composed = ". ".join(parts)
    else:
        raise DuplicateError(f"Estrategia de texto desconocida: {strategy!r}")

    composed = composed.strip()
    if not composed:
        raise DuplicateError(f"{listing.incoming_id} no produce texto que buscar")
    return composed


@dataclass(frozen=True, slots=True)
class ListingEvidence:
    """What the vector store found for one listing, before deciding anything."""

    listing: IncomingListing
    candidates: tuple[SearchHit, ...]

    @property
    def best(self) -> SearchHit | None:
        return self.candidates[0] if self.candidates else None

    @property
    def score(self) -> float:
        return self.best.native_score if self.best else 0.0

    @property
    def runner_up_score(self) -> float | None:
        return self.candidates[1].native_score if len(self.candidates) > 1 else None

    @property
    def margin(self) -> float | None:
        runner_up = self.runner_up_score
        return None if runner_up is None else self.score - runner_up


def gather_evidence(
    retriever: Retriever,
    listings: Sequence[IncomingListing],
    *,
    strategy: TextStrategy = "title_brand_color",
    candidates: int = CANDIDATES_PER_LISTING,
) -> list[ListingEvidence]:
    """Retrieve candidates for every listing through the vector store."""
    return [
        ListingEvidence(
            listing=listing,
            candidates=tuple(
                retriever.search(compose_listing(listing, strategy), top_k=candidates)
            ),
        )
        for listing in listings
    ]


def decide(evidence: ListingEvidence, *, threshold: float) -> DuplicateDecision:
    """Apply the rule to one listing's evidence.

    A positive always names the product, which ``DuplicateDecision`` enforces
    on construction — point 5 of the delivery checklist cannot be violated here
    even by mistake.
    """
    best = evidence.best
    is_duplicate = best is not None and evidence.score >= threshold
    return DuplicateDecision(
        incoming_id=evidence.listing.incoming_id,
        predicted_duplicate=is_duplicate,
        matched_product_id=best.product_id if is_duplicate and best else "",
        score=evidence.score,
        runner_up_score=evidence.runner_up_score,
    )


@dataclass(frozen=True, slots=True)
class ThresholdOutcome:
    """Confusion matrix and metrics at one candidate threshold."""

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    wrong_candidate: int = 0

    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "wrong_candidate": self.wrong_candidate,
        }


@dataclass(slots=True)
class Calibration:
    """The sweep, the chosen threshold, and why it was chosen."""

    outcomes: tuple[ThresholdOutcome, ...]
    threshold: float
    strategy: TextStrategy
    duplicate_scores: tuple[float, ...]
    new_scores: tuple[float, ...]
    notes: list[str] = field(default_factory=list)

    @property
    def separation(self) -> float:
        """Gap between the weakest duplicate and the strongest new product.

        Positive means the two classes do not overlap, so any threshold inside
        the gap classifies development perfectly. Its width is what tells us how
        much room there is for an unseen case to fall in between.
        """
        if not self.duplicate_scores or not self.new_scores:
            return 0.0
        return min(self.duplicate_scores) - max(self.new_scores)

    @property
    def best(self) -> ThresholdOutcome:
        return max(self.outcomes, key=lambda item: (item.f1, -item.false_negatives))

    def as_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "text_strategy": self.strategy,
            "separation": self.separation,
            "duplicate_scores": {
                "min": min(self.duplicate_scores) if self.duplicate_scores else 0.0,
                "max": max(self.duplicate_scores) if self.duplicate_scores else 0.0,
                "mean": fmean(self.duplicate_scores) if self.duplicate_scores else 0.0,
            },
            "new_scores": {
                "min": min(self.new_scores) if self.new_scores else 0.0,
                "max": max(self.new_scores) if self.new_scores else 0.0,
                "mean": fmean(self.new_scores) if self.new_scores else 0.0,
            },
            "chosen": next(
                (o.as_dict() for o in self.outcomes if o.threshold == self.threshold),
                None,
            ),
            "sweep": [outcome.as_dict() for outcome in self.outcomes],
            "notes": self.notes,
        }


def score_threshold(
    evidences: Sequence[ListingEvidence], threshold: float
) -> ThresholdOutcome:
    """Evaluate one threshold against labelled evidence."""
    tp = fp = tn = fn = wrong = 0
    for evidence in evidences:
        listing = evidence.listing
        if listing.is_duplicate is None:
            raise DuplicateError(
                f"{listing.incoming_id} no está etiquetado: no se puede calibrar con él"
            )
        decision = decide(evidence, threshold=threshold)
        if decision.predicted_duplicate and listing.is_duplicate:
            tp += 1
            if decision.matched_product_id != listing.reference_product_id:
                # Acierta que es duplicado pero señala otro producto. Cuenta
                # como positivo y se reporta aparte: el enunciado exige señalar
                # el product_id concreto.
                wrong += 1
        elif decision.predicted_duplicate and not listing.is_duplicate:
            fp += 1
        elif not decision.predicted_duplicate and listing.is_duplicate:
            fn += 1
        else:
            tn += 1
    return ThresholdOutcome(
        threshold=threshold,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        wrong_candidate=wrong,
    )


def calibrate(
    evidences: Sequence[ListingEvidence],
    *,
    strategy: TextStrategy = "title_brand_color",
    steps: int = 200,
) -> Calibration:
    """Sweep thresholds over labelled listings and pick one.

    When the classes separate cleanly, every threshold inside the gap scores a
    perfect F1, so F1 alone cannot choose. The midpoint of the gap is taken
    because it leaves the widest margin on both sides for cases not seen here
    — the sweep decides *the range*, and robustness decides *the point*.
    """
    if not evidences:
        raise DuplicateError("No hay evidencia con la que calibrar")

    duplicate_scores = tuple(
        e.score for e in evidences if e.listing.is_duplicate is True
    )
    new_scores = tuple(e.score for e in evidences if e.listing.is_duplicate is False)
    if not duplicate_scores or not new_scores:
        raise DuplicateError(
            "La calibración necesita casos positivos y negativos etiquetados"
        )

    low = min(min(duplicate_scores), min(new_scores))
    high = max(max(duplicate_scores), max(new_scores))
    span = high - low
    outcomes = tuple(
        score_threshold(evidences, low + span * step / steps)
        for step in range(steps + 1)
    )

    separation = min(duplicate_scores) - max(new_scores)
    notes = []
    if separation > 0:
        chosen = (min(duplicate_scores) + max(new_scores)) / 2
        notes.append(
            f"Las dos clases no se solapan: el peor duplicado puntúa "
            f"{min(duplicate_scores):.4f} y el mejor producto nuevo "
            f"{max(new_scores):.4f}. Cualquier umbral en ese hueco clasifica "
            f"el desarrollo sin error, así que F1 no puede elegir por sí solo; "
            f"se toma el punto medio ({chosen:.4f}) por dejar el margen más "
            f"ancho a ambos lados."
        )
    else:
        # Con solape, el barrido sí decide: se maximiza F1 y, a igualdad, se
        # prefiere menos falsos negativos (ver el análisis de costes).
        best = max(outcomes, key=lambda item: (item.f1, -item.false_negatives))
        chosen = best.threshold
        notes.append(
            f"Las clases se solapan ({separation:.4f}): el umbral sale de "
            f"maximizar F1, desempatando por menos falsos negativos."
        )

    return Calibration(
        outcomes=outcomes,
        threshold=chosen,
        strategy=strategy,
        duplicate_scores=duplicate_scores,
        new_scores=new_scores,
        notes=notes,
    )


def predict(
    evidences: Sequence[ListingEvidence], *, threshold: float
) -> list[DuplicateDecision]:
    """Apply a frozen threshold to unlabelled listings."""
    return [decide(evidence, threshold=threshold) for evidence in evidences]


def error_analysis(
    evidences: Sequence[ListingEvidence], *, threshold: float
) -> Mapping[str, object]:
    """Break the errors down by type, since they cost different things (RF-23).

    A **false positive** blocks a legitimate publication: the seller is held up
    and someone has to review it by hand. Annoying, visible, and recoverable.

    A **false negative** publishes a duplicate: the catalog degrades quietly,
    the same product competes with itself, and the sales signal splits between
    two listings. Cheaper to miss, far more expensive to live with.
    """
    outcome = score_threshold(evidences, threshold)
    false_positives, false_negatives, wrong = [], [], []
    for evidence in evidences:
        decision = decide(evidence, threshold=threshold)
        listing = evidence.listing
        detail = {
            "incoming_id": listing.incoming_id,
            "title": listing.title[:80],
            "score": evidence.score,
            "margin": evidence.margin,
            "matched": decision.matched_product_id,
            "expected": listing.reference_product_id or "",
        }
        if decision.predicted_duplicate and listing.is_duplicate is False:
            false_positives.append(detail)
        elif not decision.predicted_duplicate and listing.is_duplicate is True:
            false_negatives.append(detail)
        elif (
            decision.predicted_duplicate
            and listing.is_duplicate
            and decision.matched_product_id != listing.reference_product_id
        ):
            wrong.append(detail)
    return {
        "threshold": threshold,
        "metrics": outcome.as_dict(),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "wrong_candidate": wrong,
        "cost_note": (
            "Un falso positivo bloquea una publicación legítima: genera fricción "
            "con el vendedor y trabajo de revisión, pero es visible y se corrige. "
            "Un falso negativo publica un duplicado: degrada el catálogo en "
            "silencio, divide la señal de venta entre dos fichas y nadie lo "
            "detecta hasta mucho después. Ante la duda, este sistema prefiere "
            "revisar de más."
        ),
    }
