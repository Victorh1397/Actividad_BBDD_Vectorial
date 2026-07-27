"""Domain types shared by every layer, with provider semantics preserved.

The types here are deliberately strict. A ``SearchHit`` cannot exist without
declaring what its score means, and a positive ``DuplicateDecision`` cannot exist
without naming the product it matched. Making the illegal state unrepresentable
is cheaper than testing for it later (RF-01, RF-12, RF-17).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Final, Literal

ScoreKind = Literal["similarity", "distance", "relevance", "unknown"]
EventOperation = Literal["UPSERT", "DELETE"]
Profile = Literal["sample", "full"]

# Dirección de ordenación implicada por cada semántica de score. "unknown" queda
# fuera a propósito: si no sabemos qué es el número, quien lo produce debe
# declarar explícitamente cómo se ordena (P-03).
SCORE_DIRECTION: Final[dict[str, bool]] = {
    "similarity": True,
    "relevance": True,
    "distance": False,
}

UUID_PATTERN: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


class ContractError(ValueError):
    """Raised when a value violates the contract declared in specs/contracts/."""


@dataclass(frozen=True, slots=True)
class CatalogRecord:
    """One sanitized catalog product, ready to be stored as a vector point.

    ``record_id`` is the stable UUIDv5 that becomes the point ID, which is what
    makes ingestion idempotent by construction (P-08). ``product_id`` is the
    commercial identifier reported in every artifact and metric.
    """

    record_id: str
    product_id: str
    title: str
    brand: str
    color: str
    locale: str
    text: str
    catalog_version: int
    active: bool

    def __post_init__(self) -> None:
        if not UUID_PATTERN.match(self.record_id):
            raise ContractError(
                f"record_id {self.record_id!r} no cumple el contrato UUIDv5 "
                "declarado en datos/manifest.json"
            )
        if not self.product_id:
            raise ContractError("product_id no puede estar vacío")
        if not self.title:
            raise ContractError(f"title vacío en el producto {self.product_id}")
        if self.catalog_version < 1:
            raise ContractError(
                f"catalog_version debe ser >= 1, recibido {self.catalog_version}"
            )

    def payload(self) -> dict[str, Any]:
        """Return the metadata stored alongside the vector.

        ``brand`` and ``color`` travel as empty strings when absent, never as
        ``"nan"`` or ``None`` (P-07). A uniform payload is what lets the brand
        filter be a plain equality check in the database.
        """
        return {
            "product_id": self.product_id,
            "title": self.title,
            "brand": self.brand,
            "color": self.color,
            "locale": self.locale,
            "catalog_version": self.catalog_version,
            "active": self.active,
        }


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One normalized result that never loses its native score semantics.

    Two systems answering the same query may return a similarity and a distance.
    Carrying ``score_kind`` and ``higher_is_better`` alongside the raw value is
    what stops anyone from averaging or comparing them by accident (P-03).
    """

    rank: int
    record_id: str
    product_id: str
    title: str
    brand: str
    native_score: float
    score_kind: ScoreKind
    higher_is_better: bool
    color: str = ""

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ContractError(f"rank debe empezar en 1, recibido {self.rank}")
        if not self.product_id:
            raise ContractError("un resultado sin product_id no es reportable")
        expected = SCORE_DIRECTION.get(self.score_kind)
        if expected is not None and expected != self.higher_is_better:
            raise ContractError(
                f"score_kind={self.score_kind!r} implica "
                f"higher_is_better={expected}, no {self.higher_is_better}"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CatalogEvent:
    """One ordered catalog mutation.

    A deletion carries no product sheet: the source file ships only the IDs,
    because removing something does not require describing it. Modelling that
    asymmetry here means the applier cannot accidentally index an empty record.

    The file only distinguishes ``UPSERT`` from ``DELETE``. Whether an upsert
    inserts or updates depends on the live collection, so that classification
    belongs to the applier, not to this type (RF-16).
    """

    sequence: int
    event_id: str
    operation: EventOperation
    record_id: str
    product_id: str
    record: CatalogRecord | None = None

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ContractError(f"sequence debe ser >= 1, recibido {self.sequence}")
        if self.operation not in ("UPSERT", "DELETE"):
            raise ContractError(f"operación desconocida: {self.operation!r}")
        if not UUID_PATTERN.match(self.record_id):
            raise ContractError(
                f"{self.event_id}: record_id {self.record_id!r} no cumple el "
                "contrato UUIDv5"
            )
        if not self.product_id:
            raise ContractError(f"{self.event_id}: product_id no puede estar vacío")
        if self.operation == "UPSERT" and self.record is None:
            raise ContractError(
                f"{self.event_id}: un UPSERT necesita la ficha completa del producto"
            )
        if self.operation == "DELETE" and self.record is not None:
            raise ContractError(
                f"{self.event_id}: un DELETE opera sobre un ID, no sobre una ficha"
            )
        if self.record is not None and self.record.record_id != self.record_id:
            raise ContractError(
                f"{self.event_id}: la ficha describe {self.record.record_id} "
                f"pero el evento apunta a {self.record_id}"
            )

    @property
    def is_deletion(self) -> bool:
        return self.operation == "DELETE"

    def require_record(self) -> CatalogRecord:
        """Return the sheet, failing loudly if the operation has none."""
        if self.record is None:
            raise ContractError(
                f"{self.event_id}: la operación {self.operation} no aporta ficha"
            )
        return self.record


@dataclass(frozen=True, slots=True)
class IncomingListing:
    """A candidate product submitted for publication.

    Development cases carry their label; evaluation cases do not. Keeping both
    in one type with optional labels avoids two near-identical classes and makes
    it obvious when a label is being read that should not exist (P-04).
    """

    incoming_id: str
    title: str
    brand: str
    color: str
    text: str
    is_duplicate: bool | None = None
    reference_product_id: str | None = None

    def __post_init__(self) -> None:
        if not self.incoming_id:
            raise ContractError("incoming_id no puede estar vacío")
        if self.is_duplicate is False and self.reference_product_id:
            raise ContractError(
                f"{self.incoming_id}: un caso etiquetado como no duplicado "
                "no puede declarar producto de referencia"
            )
        if self.is_duplicate is True and not self.reference_product_id:
            raise ContractError(
                f"{self.incoming_id}: un duplicado etiquetado debe declarar "
                "su producto de referencia"
            )

    @property
    def is_labelled(self) -> bool:
        """Whether this case belongs to the development set."""
        return self.is_duplicate is not None


@dataclass(frozen=True, slots=True)
class DuplicateDecision:
    """The verdict for one incoming listing.

    Point 5 of the delivery checklist is encoded in the type itself: a positive
    prediction that does not name a product simply cannot be constructed.
    """

    incoming_id: str
    predicted_duplicate: bool
    matched_product_id: str
    score: float
    runner_up_score: float | None = None

    def __post_init__(self) -> None:
        if self.predicted_duplicate and not self.matched_product_id:
            raise ContractError(
                f"{self.incoming_id}: una predicción positiva debe señalar el "
                "product_id concreto que se considera duplicado"
            )
        if not self.predicted_duplicate and self.matched_product_id:
            raise ContractError(
                f"{self.incoming_id}: una predicción negativa no propone candidato"
            )

    @property
    def margin(self) -> float | None:
        """Distance to the runner-up, the signal that separates a near-tie."""
        if self.runner_up_score is None:
            return None
        return self.score - self.runner_up_score

    def as_row(self) -> dict[str, Any]:
        """Return the row shape declared in resultados_duplicados.schema.json."""
        return {
            "incoming_id": self.incoming_id,
            "predicted_duplicate": self.predicted_duplicate,
            "matched_product_id": self.matched_product_id,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """One query from any of the three workloads."""

    query_id: str
    text: str
    query_type: str = ""
    brand: str | None = None
    numeric_id: int | None = None

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ContractError("query_id no puede estar vacío")
        if not self.text.strip():
            raise ContractError(f"{self.query_id}: el texto de consulta está vacío")


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """A loaded catalog profile with its lookups already resolved."""

    profile: Profile
    records: tuple[CatalogRecord, ...]
    by_record_id: dict[str, CatalogRecord] = field(repr=False, default_factory=dict)
    by_product_id: dict[str, CatalogRecord] = field(repr=False, default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(record.text for record in self.records)

    @property
    def product_ids(self) -> tuple[str, ...]:
        return tuple(record.product_id for record in self.records)
