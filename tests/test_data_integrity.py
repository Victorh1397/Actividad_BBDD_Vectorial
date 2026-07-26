"""Dataset integrity and the single sanitization rule (RF-03, RF-07)."""

from __future__ import annotations

import math

import pytest

from aurum_market.data import (
    EXPECTED_FILES,
    clean_scalar,
    load_manifest,
    verify_data_integrity,
)


class TestManifest:
    def test_manifest_declares_the_expected_snapshot(self) -> None:
        manifest = load_manifest()
        assert manifest["snapshot_id"] == "aurum-market-vector-search-evaluation-v1"

    def test_manifest_counts_match_the_statement(self) -> None:
        counts = load_manifest()["counts"]
        assert counts["catalog_records"] == 15_000
        assert counts["sample_records"] == 1_500
        assert counts["development_queries"] == 8
        assert counts["evaluation_queries"] == 12
        assert counts["filtered_queries"] == 4
        assert counts["catalog_events"] == 24
        assert counts["duplicate_development_cases"] == 14
        assert counts["duplicate_evaluation_cases"] == 14

    def test_relevance_mapping_is_the_esci_one(self) -> None:
        mapping = load_manifest()["selection"]["relevance_mapping"]
        assert mapping == {"E": 3, "S": 2, "C": 1, "I": 0}


class TestFileIntegrity:
    def test_every_expected_file_is_present_and_matches_its_checksum(self) -> None:
        checks = verify_data_integrity(verify_checksums=True)
        assert len(checks) == len(EXPECTED_FILES)
        broken = [check for check in checks if not check.ok]
        assert not broken, f"Ficheros con integridad rota: {[c.name for c in broken]}"

    def test_reserved_teacher_data_is_not_expected(self) -> None:
        """Las relevancias del conjunto ciego no forman parte de la entrega."""
        assert not any("profesorado" in name for name in EXPECTED_FILES)


class TestScalarSanitization:
    """P-07: a missing value is empty, never the literal string "nan"."""

    @pytest.mark.parametrize(
        "value",
        [None, float("nan"), "nan", "NaN", "None", "null", "<NA>", "", "   "],
    )
    def test_missing_values_never_render_as_nan(self, value: object) -> None:
        assert clean_scalar(value) == ""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("  NIKE  ", "NIKE"),
            ("Negro (Black/White 011)", "Negro (Black/White 011)"),
            (2, "2"),
            (True, "True"),
        ],
    )
    def test_real_values_survive_untouched(self, value: object, expected: str) -> None:
        assert clean_scalar(value) == expected

    def test_a_brand_named_like_a_null_keyword_is_still_dropped(self) -> None:
        """Decisión consciente: preferimos perder una marca improbable a inyectar "nan"."""
        assert clean_scalar("None") == ""

    def test_nan_float_is_detected_by_value_not_by_string(self) -> None:
        value = float("nan")
        assert math.isnan(value)
        assert clean_scalar(value) == ""
