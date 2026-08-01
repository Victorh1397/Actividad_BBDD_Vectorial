"""Strategies for composing the text that gets encoded (RF-03, RF-06).

Which fields make up the encoded text is a real decision, not plumbing. The
catalog ships a ``text`` field that averages 1.309 characters and reaches 3.000,
much of it keyword stuffing repeated for search-engine purposes. Multilingual
E5 truncates at 512 tokens (~2.048 characters), so **27,2 % of products lose
their tail** — and what survives may be noise rather than signal.

That is the hypothesis experiments E1 and E2 are designed to test.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final, Literal, get_args

from .contracts import CatalogRecord

TextStrategy = Literal["raw_text", "title_brand_color", "title_only"]

TEXT_STRATEGIES: Final[tuple[TextStrategy, ...]] = get_args(TextStrategy)

STRATEGY_DESCRIPTIONS: Final[dict[TextStrategy, str]] = {
    "raw_text": (
        "El campo `text` del catálogo, sin tocar. Es la representación de partida "
        "y la que más contexto aporta, pero también la que arrastra el ruido de "
        "las descripciones cargadas de palabras clave."
    ),
    "title_brand_color": (
        "Título más los metadatos disponibles, con etiquetas explícitas. Cabe "
        "siempre dentro del límite del modelo y concentra la señal identificativa "
        "del producto."
    ),
    "title_only": (
        "Solo el título. Sirve de control: mide cuánto aportan realmente marca y "
        "color frente a la denominación del producto."
    ),
}


class TextCompositionError(ValueError):
    """Raised when a strategy cannot produce usable text for a record."""


def compose(record: CatalogRecord, strategy: TextStrategy = "raw_text") -> str:
    """Return the text that represents ``record`` under ``strategy``.

    Absent metadata is skipped rather than rendered as an empty label: a product
    with no colour must not be encoded as "Color: ", which would teach the model
    a pattern that means nothing (P-07).
    """
    if strategy not in TEXT_STRATEGIES:
        known = ", ".join(TEXT_STRATEGIES)
        raise TextCompositionError(
            f"Estrategia de texto desconocida: {strategy!r}. Usa una de: {known}"
        )

    if strategy == "raw_text":
        composed = record.text or record.title
    elif strategy == "title_only":
        composed = record.title
    else:
        parts = [record.title]
        if record.brand:
            parts.append(f"Marca: {record.brand}")
        if record.color:
            parts.append(f"Color: {record.color}")
        composed = ". ".join(parts)

    composed = composed.strip()
    if not composed:
        raise TextCompositionError(
            f"El producto {record.product_id} no produce texto con {strategy!r}"
        )
    return composed


def compose_all(
    records: Iterable[CatalogRecord], strategy: TextStrategy = "raw_text"
) -> tuple[str, ...]:
    """Compose every record, preserving order so vectors stay aligned with IDs."""
    return tuple(compose(record, strategy) for record in records)


def describe(strategy: TextStrategy) -> str:
    """Return the human-readable rationale, for the experiment report."""
    if strategy not in STRATEGY_DESCRIPTIONS:
        raise TextCompositionError(f"Estrategia de texto desconocida: {strategy!r}")
    return STRATEGY_DESCRIPTIONS[strategy]
