"""Settings must fail loudly at start-up, never halfway through a run (RF-18)."""

from __future__ import annotations

import pytest

from aurum_market.config import (
    RESOURCE_PREFIX,
    ConfigurationError,
    HnswSettings,
    Settings,
    validate_resource_name,
)


def make_settings(**overrides: object) -> Settings:
    """Build a valid Settings, overriding only what a test cares about."""
    defaults: dict[str, object] = {
        "embedding_model": "intfloat/multilingual-e5-small",
        "qdrant_url": "http://localhost:6333",
        "qdrant_api_key": None,
        "qdrant_collection": f"{RESOURCE_PREFIX}-catalogo",
        "batch_size": 256,
        "allow_reset": False,
        "confirm_cleanup": "",
    }
    return Settings(**(defaults | overrides))  # type: ignore[arg-type]


class TestValidation:
    def test_accepts_a_sane_configuration(self) -> None:
        settings = make_settings()
        assert settings.embedding_dimension == 384

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("embedding_model", "   "),
            ("qdrant_url", "localhost:6333"),
            ("batch_size", 0),
            ("batch_size", -1),
        ],
    )
    def test_rejects_invalid_settings(self, field: str, value: object) -> None:
        with pytest.raises(ConfigurationError):
            make_settings(**{field: value})

    def test_unknown_model_has_no_silent_dimension(self) -> None:
        settings = make_settings(embedding_model="acme/some-unknown-model")
        with pytest.raises(ConfigurationError, match="Dimensión desconocida"):
            _ = settings.embedding_dimension

    @pytest.mark.parametrize(
        "hnsw",
        [
            {"m": 1},
            {"ef_construct": 3},
            {"ef_search": 0},
        ],
    )
    def test_rejects_impossible_hnsw_parameters(self, hnsw: dict[str, int]) -> None:
        with pytest.raises(ConfigurationError):
            HnswSettings(**hnsw)


class TestResourceProtection:
    """P-11: nothing outside the activity namespace can ever be touched."""

    def test_accepts_names_inside_the_protected_prefix(self) -> None:
        assert validate_resource_name(f"{RESOURCE_PREFIX}-catalogo")

    @pytest.mark.parametrize(
        "name",
        ["produccion-catalogo", "catalogo", "aurum", "otro-aurum-market"],
    )
    def test_rejects_names_outside_the_protected_prefix(self, name: str) -> None:
        with pytest.raises(ConfigurationError, match="prefijo protegido"):
            validate_resource_name(name)

    def test_a_collection_outside_the_prefix_cannot_be_configured(self) -> None:
        with pytest.raises(ConfigurationError):
            make_settings(qdrant_collection="produccion-catalogo")
