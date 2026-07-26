"""Operational safety. Closes RF-18 and point 7 of "Antes de entregar"."""

from __future__ import annotations

import pytest

from aurum_market.config import PROJECT_ROOT, RESOURCE_PREFIX

from .test_config import make_settings

COLLECTION = f"{RESOURCE_PREFIX}-catalogo"


class TestCleanupAuthorization:
    """A destructive operation needs BOTH the flag and the exact name (P-11)."""

    def test_cleanup_is_disabled_by_default(self) -> None:
        settings = make_settings()
        assert settings.cleanup_authorized(COLLECTION) is False

    def test_permission_alone_is_not_enough(self) -> None:
        settings = make_settings(allow_reset=True, confirm_cleanup="")
        assert settings.cleanup_authorized(COLLECTION) is False

    def test_confirmation_alone_is_not_enough(self) -> None:
        settings = make_settings(allow_reset=False, confirm_cleanup=COLLECTION)
        assert settings.cleanup_authorized(COLLECTION) is False

    def test_both_conditions_authorize_the_cleanup(self) -> None:
        settings = make_settings(allow_reset=True, confirm_cleanup=COLLECTION)
        assert settings.cleanup_authorized(COLLECTION) is True

    def test_confirmation_must_name_the_very_same_resource(self) -> None:
        settings = make_settings(
            allow_reset=True, confirm_cleanup=f"{RESOURCE_PREFIX}-otra-coleccion"
        )
        assert settings.cleanup_authorized(COLLECTION) is False


class TestRepositoryHygiene:
    """Point 7: no keys, no volumes, no reserved data in the repository."""

    def test_repository_has_no_secrets_or_reserved_data(self) -> None:
        assert not (PROJECT_ROOT / ".env").is_relative_to(PROJECT_ROOT / "datos")
        # Los datos reservados a la corrección nunca deben estar presentes.
        assert not (PROJECT_ROOT / "datos" / "profesorado").exists()
        # .env queda fuera del control de versiones.
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert ".env" in gitignore
        assert "datos/profesorado/" in gitignore

    def test_env_example_carries_no_secret_values(self) -> None:
        template = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        for line in template.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            key, _, value = stripped.partition("=")
            if key.strip().endswith(("API_KEY", "TOKEN", "PASSWORD", "SECRET")):
                assert value.strip() == "", f"{key} no puede traer un valor por defecto"

    def test_cleanup_is_off_in_the_template(self) -> None:
        template = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        assert "AURUM_ALLOW_RESET=false" in template
        assert "AURUM_CONFIRM_CLEANUP=\n" in template or template.rstrip().endswith(
            "AURUM_CONFIRM_CLEANUP="
        )


class TestDataIntegrityIsEnforced:
    def test_gitattributes_protects_data_bytes(self) -> None:
        """Sin esto, autocrlf reescribe los CSV y los checksums fallan (P-06)."""
        path = PROJECT_ROOT / ".gitattributes"
        assert path.is_file(), (
            "Falta .gitattributes: los checksums no sobrevivirían a un clon"
        )
        assert "datos/** -text" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("forbidden", ["qdrant_storage", ".qdrant"])
def test_no_engine_volumes_are_committed(forbidden: str) -> None:
    assert not (PROJECT_ROOT / forbidden).exists()
