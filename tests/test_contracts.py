"""Domain types must make illegal states unrepresentable (RF-01, RF-12, RF-17).

Every dataclass is also validated against the JSON schema that declares it in
specs/contracts/, so the code and the specification cannot drift apart.
"""

from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from aurum_market.config import CONTRACTS_DIRECTORY
from aurum_market.contracts import (
    CatalogEvent,
    CatalogRecord,
    ContractError,
    DuplicateDecision,
    IncomingListing,
    RetrievalQuery,
    SearchHit,
)

VALID_UUID = "000bd6e8-a995-56d0-ba03-559885ccef39"


def schema_validator(name: str) -> Draft202012Validator:
    """Return a validator for one of the declared contracts."""
    path = CONTRACTS_DIRECTORY / f"{name}.schema.json"
    return Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))


def make_record(**overrides: object) -> CatalogRecord:
    defaults: dict[str, object] = {
        "record_id": VALID_UUID,
        "product_id": "B0818K237B",
        "title": "Vestido largo de fiesta",
        "brand": "KanLin1986-Ropa",
        "color": "Negro",
        "locale": "es",
        "text": "Vestido largo de fiesta. Marca: KanLin1986-Ropa. Color: Negro.",
        "catalog_version": 1,
        "active": True,
    }
    return CatalogRecord(**(defaults | overrides))  # type: ignore[arg-type]


def make_hit(**overrides: object) -> SearchHit:
    defaults: dict[str, object] = {
        "rank": 1,
        "record_id": VALID_UUID,
        "product_id": "B0818K237B",
        "title": "Vestido largo de fiesta",
        "brand": "KanLin1986-Ropa",
        "native_score": 0.87,
        "score_kind": "similarity",
        "higher_is_better": True,
    }
    return SearchHit(**(defaults | overrides))  # type: ignore[arg-type]


class TestCatalogRecord:
    def test_a_valid_record_matches_its_json_contract(self) -> None:
        record = make_record()
        payload = {
            "record_id": record.record_id,
            "product_id": record.product_id,
            "title": record.title,
            "brand": record.brand,
            "color": record.color,
            "locale": record.locale,
            "text": record.text,
            "catalog_version": record.catalog_version,
            "active": record.active,
        }
        schema_validator("catalog_record").validate(payload)

    @pytest.mark.parametrize(
        "bad_id",
        [
            "no-es-un-uuid",
            "000bd6e8a99556d0ba03559885ccef39",
            # UUIDv4: la versión debe ser 5, porque el contrato es determinista.
            "000bd6e8-a995-46d0-ba03-559885ccef39",
            "",
        ],
    )
    def test_rejects_ids_outside_the_uuid5_contract(self, bad_id: str) -> None:
        with pytest.raises(ContractError, match="UUIDv5"):
            make_record(record_id=bad_id)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("product_id", ""), ("title", ""), ("catalog_version", 0)],
    )
    def test_rejects_unreportable_records(self, field: str, value: object) -> None:
        with pytest.raises(ContractError):
            make_record(**{field: value})

    def test_missing_metadata_travels_as_empty_string(self) -> None:
        """P-07: el payload nunca lleva None ni "nan" a la base de datos."""
        payload = make_record(brand="", color="").payload()
        assert payload["brand"] == ""
        assert payload["color"] == ""
        assert all(value is not None for value in payload.values())

    def test_payload_carries_the_product_id(self) -> None:
        """Sin product_id en el payload no se pueden escribir los artefactos."""
        assert make_record().payload()["product_id"] == "B0818K237B"


class TestSearchHit:
    def test_a_valid_hit_matches_its_json_contract(self) -> None:
        schema_validator("search_hit").validate(make_hit().as_dict())

    def test_search_hit_contract_exposes_the_minimum_fields(self) -> None:
        """RF-01: product_id, posición, título, metadatos y score."""
        hit = make_hit()
        for attribute in ("product_id", "rank", "title", "brand", "native_score"):
            assert hasattr(hit, attribute)

    def test_score_semantics_are_declared(self) -> None:
        hit = make_hit()
        assert hit.score_kind == "similarity"
        assert hit.higher_is_better is True

    def test_a_distance_cannot_claim_higher_is_better(self) -> None:
        """P-03: el error clásico de tratar una distancia como una similitud."""
        with pytest.raises(ContractError, match="higher_is_better"):
            make_hit(score_kind="distance", higher_is_better=True)

    def test_a_distance_orders_the_other_way_round(self) -> None:
        hit = make_hit(score_kind="distance", higher_is_better=False, native_score=0.13)
        assert hit.higher_is_better is False

    def test_an_unknown_score_must_declare_its_direction_freely(self) -> None:
        """Si no sabemos qué es el número, quien lo produce decide la dirección."""
        assert (
            make_hit(score_kind="unknown", higher_is_better=False).higher_is_better
            is False
        )
        assert (
            make_hit(score_kind="unknown", higher_is_better=True).higher_is_better
            is True
        )

    @pytest.mark.parametrize("rank", [0, -1])
    def test_ranks_start_at_one(self, rank: int) -> None:
        with pytest.raises(ContractError, match="rank"):
            make_hit(rank=rank)


class TestCatalogEvent:
    def test_exposes_the_identifiers_of_its_record(self) -> None:
        event = CatalogEvent(
            sequence=1, event_id="EVT-001", operation="UPSERT", record=make_record()
        )
        assert event.record_id == VALID_UUID
        assert event.product_id == "B0818K237B"

    def test_rejects_unknown_operations(self) -> None:
        with pytest.raises(ContractError, match="operación desconocida"):
            CatalogEvent(
                sequence=1,
                event_id="EVT-001",
                operation="MERGE",  # type: ignore[arg-type]
                record=make_record(),
            )

    def test_sequence_starts_at_one(self) -> None:
        with pytest.raises(ContractError, match="sequence"):
            CatalogEvent(
                sequence=0, event_id="EVT-001", operation="DELETE", record=make_record()
            )


class TestIncomingListing:
    def test_development_cases_are_labelled(self) -> None:
        listing = IncomingListing(
            incoming_id="DEV-DUP-001",
            title="NIKE Legasee Legging",
            brand="NIKE",
            color="Negro",
            text="NIKE Legasee Legging",
            is_duplicate=True,
            reference_product_id="B000G3T55M",
        )
        assert listing.is_labelled is True

    def test_evaluation_cases_carry_no_label(self) -> None:
        listing = IncomingListing(
            incoming_id="EVAL-DUP-001",
            title="Bolsa para portátil",
            brand="Joyfeel buy",
            color="Pink",
            text="Bolsa para portátil",
        )
        assert listing.is_labelled is False
        assert listing.is_duplicate is None

    def test_a_labelled_duplicate_must_name_its_reference(self) -> None:
        with pytest.raises(ContractError, match="producto de referencia"):
            IncomingListing(
                incoming_id="DEV-DUP-001",
                title="x",
                brand="",
                color="",
                text="x",
                is_duplicate=True,
                reference_product_id=None,
            )

    def test_a_labelled_non_duplicate_cannot_name_a_reference(self) -> None:
        with pytest.raises(ContractError, match="no duplicado"):
            IncomingListing(
                incoming_id="DEV-DUP-008",
                title="x",
                brand="",
                color="",
                text="x",
                is_duplicate=False,
                reference_product_id="B000G3T55M",
            )


class TestDuplicateDecision:
    def test_positive_prediction_names_a_candidate(self) -> None:
        """Punto 5 de "Antes de entregar", codificado en el tipo."""
        decision = DuplicateDecision(
            incoming_id="EVAL-DUP-001",
            predicted_duplicate=True,
            matched_product_id="B000G3T55M",
            score=0.94,
        )
        assert decision.matched_product_id

    def test_a_positive_without_candidate_cannot_be_built(self) -> None:
        with pytest.raises(ContractError, match="product_id concreto"):
            DuplicateDecision(
                incoming_id="EVAL-DUP-001",
                predicted_duplicate=True,
                matched_product_id="",
                score=0.94,
            )

    def test_a_negative_does_not_propose_a_candidate(self) -> None:
        with pytest.raises(ContractError, match="no propone candidato"):
            DuplicateDecision(
                incoming_id="EVAL-DUP-002",
                predicted_duplicate=False,
                matched_product_id="B000G3T55M",
                score=0.31,
            )

    def test_margin_is_none_without_a_runner_up(self) -> None:
        decision = DuplicateDecision("EVAL-DUP-003", False, "", 0.42)
        assert decision.margin is None

    def test_margin_separates_a_near_tie_from_a_clear_match(self) -> None:
        clear = DuplicateDecision("A", True, "B000G3T55M", 0.95, runner_up_score=0.40)
        tie = DuplicateDecision("B", True, "B000G3T55M", 0.95, runner_up_score=0.94)
        assert clear.margin == pytest.approx(0.55)
        assert tie.margin == pytest.approx(0.01)

    def test_rows_validate_against_the_delivery_contract(self) -> None:
        rows = [
            DuplicateDecision("EVAL-DUP-001", True, "B000G3T55M", 0.94).as_row(),
            DuplicateDecision("EVAL-DUP-002", False, "", 0.31).as_row(),
        ]
        validator = schema_validator("resultados_duplicados")
        # El esquema completo exige 14 filas; aquí validamos la forma de cada una.
        item_validator = Draft202012Validator(validator.schema["items"])
        for row in rows:
            item_validator.validate(row)


class TestRetrievalQuery:
    def test_carries_an_optional_brand_constraint(self) -> None:
        query = RetrievalQuery(
            query_id="FILTER-001",
            text="herramienta inalámbrica para perforar",
            brand="Einhell",
        )
        assert query.brand == "Einhell"

    @pytest.mark.parametrize("text", ["", "   "])
    def test_rejects_empty_query_text(self, text: str) -> None:
        with pytest.raises(ContractError, match="texto de consulta"):
            RetrievalQuery(query_id="DEV-1", text=text)
