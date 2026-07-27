"""Catalog loading, sanitization and the UUIDv5 ID contract (RF-03, RF-07)."""

from __future__ import annotations

import pytest

from aurum_market.contracts import CatalogRecord
from aurum_market.data import (
    DataIntegrityError,
    clean_flag,
    clean_version,
    expected_record_id,
    load_catalog,
    read_catalog_frame,
    record_from_row,
)


@pytest.fixture(scope="module")
def sample_catalog():
    """The 1.500-record development profile, loaded once for the module."""
    return load_catalog("sample")


class TestProfiles:
    def test_sample_profile_loads_the_expected_count(self, sample_catalog) -> None:
        assert len(sample_catalog) == 1_500
        assert sample_catalog.profile == "sample"

    @pytest.mark.slow
    def test_full_profile_loads_the_evaluated_catalog(self) -> None:
        catalog = load_catalog("full")
        assert len(catalog) == 15_000

    def test_an_unknown_profile_is_rejected(self) -> None:
        with pytest.raises(DataIntegrityError, match="Perfil desconocido"):
            read_catalog_frame("produccion")  # type: ignore[arg-type]


class TestIdContract:
    def test_record_id_follows_the_uuid5_contract(self, sample_catalog) -> None:
        """Cada record_id debe ser uuid5(namespace, product_id) (RF-07)."""
        for record in sample_catalog.records:
            assert record.record_id == expected_record_id(record.product_id)

    def test_the_contract_is_deterministic(self) -> None:
        assert expected_record_id("B0818K237B") == expected_record_id("B0818K237B")
        assert (
            expected_record_id("B0818K237B") == "000bd6e8-a995-56d0-ba03-559885ccef39"
        )

    def test_different_products_get_different_ids(self) -> None:
        assert expected_record_id("B0818K237B") != expected_record_id("B086YX9RK5")

    def test_a_blank_product_id_has_no_record_id(self) -> None:
        with pytest.raises(DataIntegrityError, match="product_id"):
            expected_record_id("   ")

    def test_ids_are_unique_across_the_catalog(self, sample_catalog) -> None:
        """Sin unicidad, el upsert idempotente perdería productos (P-08)."""
        assert len(sample_catalog.by_record_id) == len(sample_catalog)
        assert len(sample_catalog.by_product_id) == len(sample_catalog)


class TestSanitization:
    def test_absent_metadata_becomes_empty_string(self, sample_catalog) -> None:
        """La muestra trae 44 marcas y 549 colores ausentes: ninguno es "nan"."""
        forbidden = {"nan", "none", "null", "<na>"}
        for record in sample_catalog.records:
            assert record.brand.lower() not in forbidden
            assert record.color.lower() not in forbidden

    def test_some_metadata_really_is_absent(self, sample_catalog) -> None:
        """Comprobación de que el test anterior no pasa por vacuidad."""
        missing_brand = sum(1 for r in sample_catalog.records if r.brand == "")
        missing_color = sum(1 for r in sample_catalog.records if r.color == "")
        assert missing_brand == 44
        assert missing_color == 549

    def test_every_record_keeps_its_essential_fields(self, sample_catalog) -> None:
        for record in sample_catalog.records:
            assert record.product_id
            assert record.title
            assert record.text
            assert record.locale == "es"

    def test_row_sanitization_handles_pandas_missing_values(self) -> None:
        record = record_from_row(
            {
                "record_id": "000bd6e8-a995-56d0-ba03-559885ccef39",
                "product_id": "B0818K237B",
                "title": "Vestido largo",
                "brand": float("nan"),
                "color": None,
                "locale": "es",
                "text": "Vestido largo",
                "catalog_version": "1",
                "active": "True",
            }
        )
        assert record.brand == ""
        assert record.color == ""
        assert record.catalog_version == 1
        assert record.active is True


class TestScalarCoercion:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(True, True), ("True", True), ("true", True), ("1", True), (1, True)],
    )
    def test_truthy_flags(self, value: object, expected: bool) -> None:
        assert clean_flag(value) is expected

    @pytest.mark.parametrize("value", [False, "False", "false", "0", "", None])
    def test_falsy_flags(self, value: object) -> None:
        assert clean_flag(value) is False

    def test_an_unrecognized_flag_is_never_guessed(self) -> None:
        with pytest.raises(DataIntegrityError, match="booleano"):
            clean_flag("quizá")

    @pytest.mark.parametrize(("value", "expected"), [("1", 1), (2, 2), ("2.0", 2)])
    def test_versions_are_integers(self, value: object, expected: int) -> None:
        assert clean_version(value) == expected

    def test_a_non_numeric_version_is_rejected(self) -> None:
        with pytest.raises(DataIntegrityError, match="catalog_version"):
            clean_version("v2")


class TestSnapshotLookups:
    def test_lookups_resolve_to_the_same_record(self, sample_catalog) -> None:
        record = sample_catalog.records[0]
        assert sample_catalog.by_record_id[record.record_id] is record
        assert sample_catalog.by_product_id[record.product_id] is record

    def test_snapshot_exposes_texts_and_ids_aligned(self, sample_catalog) -> None:
        assert len(sample_catalog.texts) == len(sample_catalog)
        assert len(sample_catalog.product_ids) == len(sample_catalog)
        assert sample_catalog.product_ids[0] == sample_catalog.records[0].product_id


class TestIntegrityFailures:
    def test_duplicate_record_ids_are_rejected(self, tmp_path, sample_catalog) -> None:
        """Un catálogo con IDs repetidos rompería la idempotencia en silencio."""
        from aurum_market.data import _validate_uniqueness

        duplicated: tuple[CatalogRecord, ...] = (
            sample_catalog.records[0],
            sample_catalog.records[0],
        )
        with pytest.raises(DataIntegrityError, match="record_id duplicados"):
            _validate_uniqueness(duplicated)
