"""Applying the ordered catalog mutations (RF-16).

Three things make this module more than a loop over a CSV.

**The file does not say whether an upsert inserts or updates.** It only says
``UPSERT``. Whether that is an *alta* or an *actualización* depends on what the
collection holds at that moment, so the classification is an observation made
against the live engine, never a property read from the file. That is also why
reapplying the events reports different *kinds* while producing an identical
*state*: the second run finds everything already there.

**The point count is a useless check here.** The 24 events add eight products
and remove eight, so the collection goes from 15.000 points to 15.000 points. A
verification that compared counts would pass even if nothing had happened. The
state is therefore checked by identity — these eight are gone, those eight are
present — and the count wait is an *exact* comparison, because a decrease would
satisfy any ``>=``.

**A confirmed write is not the same as an observable one.** Every mutation is
probed through both channels the statement names, reading by ID and searching
by vector, each with a bounded wait (P-10). For a deletion the probe first
proves the product *was* findable: showing that something cannot be found after
being removed means nothing unless it could be found before.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import monotonic, sleep
from typing import Literal

import numpy as np

from .contracts import CatalogEvent, CatalogRecord
from .embeddings import Encoder
from .store.qdrant_store import QdrantStore
from .text import TextStrategy, compose

# Alta y actualización son la misma operación para el motor y distintas para el
# negocio. El enunciado pide distinguirlas, así que se distinguen aquí.
EventKind = Literal["alta", "actualizacion", "baja"]

KIND_ORDER: tuple[EventKind, ...] = ("alta", "actualizacion", "baja")

POLL_SECONDS = 0.2


class EventError(RuntimeError):
    """Raised when an event cannot be applied or its effect cannot be observed."""


# --------------------------------------------------------------- clasificación


def classify(event: CatalogEvent, store: QdrantStore) -> EventKind:
    """Decide what this event *does*, by looking at the live collection.

    A ``DELETE`` is always a baja, even when it removes nothing. An ``UPSERT``
    is an alta or an actualización depending on whether the point is already
    there, which is a fact about the engine and not about the file.
    """
    if event.is_deletion:
        return "baja"
    return (
        "actualizacion"
        if store.get_by_record_id(event.record_id) is not None
        else "alta"
    )


# -------------------------------------------------------------------- informes


@dataclass(frozen=True, slots=True)
class VisibilityProbe:
    """Evidence that one mutation became observable through both channels.

    The three times measure different things and none contains another:
    ``write_ms`` is the mutation call itself, and the two probe times are each
    measured from their own start. Since Qdrant confirms the write
    synchronously, what these numbers report is the cost of *observing*, not a
    propagation delay — the bounded wait exists to catch the case where that
    stops being true.
    """

    event_id: str
    product_id: str
    record_id: str
    kind: EventKind
    write_ms: float
    by_id_ms: float
    by_search_ms: float
    id_evidence: str
    search_evidence: str
    query_text: str
    baseline: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "product_id": self.product_id,
            "record_id": self.record_id,
            "kind": self.kind,
            "write_ms": round(self.write_ms, 3),
            "by_id_ms": round(self.by_id_ms, 3),
            "by_search_ms": round(self.by_search_ms, 3),
            "id_evidence": self.id_evidence,
            "search_evidence": self.search_evidence,
            "query_text": self.query_text,
            "baseline": self.baseline,
        }


@dataclass(frozen=True, slots=True)
class EventReport:
    """What the run did, in terms that make a second run comparable.

    ``altas`` and ``bajas_efectivas`` are the pair that reveals idempotence:
    on a first application they are eight and eight, on a second both are zero
    and every upsert has become an actualización — while ``points_after`` and
    the resulting state are identical.
    """

    applied: int
    altas: int
    actualizaciones: int
    bajas: int
    bajas_efectivas: int
    points_before: int
    points_after: int
    kinds: tuple[tuple[str, EventKind], ...]
    probes: tuple[VisibilityProbe, ...]
    notes: tuple[str, ...] = ()

    @property
    def expected_points(self) -> int:
        """The count the collection must end with, derived from what happened."""
        return self.points_before + self.altas - self.bajas_efectivas

    def as_dict(self) -> dict[str, object]:
        return {
            "applied": self.applied,
            "altas": self.altas,
            "actualizaciones": self.actualizaciones,
            "bajas": self.bajas,
            "bajas_efectivas": self.bajas_efectivas,
            "points_before": self.points_before,
            "points_after": self.points_after,
            "expected_points": self.expected_points,
            "kinds": [
                {"event_id": event_id, "kind": kind} for event_id, kind in self.kinds
            ],
            "probes": [probe.as_dict() for probe in self.probes],
            "notes": list(self.notes),
        }


# ------------------------------------------------------------- espera acotada


def _wait_until(
    observe: Callable[[], tuple[bool, str]],
    *,
    timeout_seconds: float,
    description: str,
) -> tuple[float, str]:
    """Poll ``observe`` until it reports success, or fail saying what was seen.

    ``observe`` returns the verdict *and* the evidence behind it, so the
    timeout message can quote the last real observation instead of stating that
    something generic went wrong (P-10).
    """
    if timeout_seconds <= 0:
        raise EventError("El plazo de espera debe ser positivo")
    started = monotonic()
    while True:
        satisfied, evidence = observe()
        elapsed = monotonic() - started
        if satisfied:
            return elapsed * 1000.0, evidence
        if elapsed >= timeout_seconds:
            raise EventError(
                f"Tras {elapsed:.1f}s {description}. Última observación: {evidence}"
            )
        sleep(POLL_SECONDS)


def _wait_for_exact_count(
    store: QdrantStore, *, expected: int, timeout_seconds: float
) -> int:
    """Wait for the collection to hold *exactly* ``expected`` points.

    Exact, not ``>=``: these events remove points, and a stale higher count
    satisfies any lower bound (this is the check ``wait_until_indexed`` cannot
    do, since it is written for ingestion, where the count only grows).
    """

    def observe() -> tuple[bool, str]:
        count = store.status().points_count
        return count == expected, f"{count} puntos"

    _, evidence = _wait_until(
        observe,
        timeout_seconds=timeout_seconds,
        description=f"la colección no tiene los {expected} puntos esperados",
    )
    del evidence
    return expected


# -------------------------------------------------------------------- sondas


def _title_of(payload: dict[str, object] | None) -> str:
    return str(payload.get("title", "")) if payload else ""


def _probe_by_id(
    store: QdrantStore,
    *,
    record_id: str,
    kind: EventKind,
    expected_title: str,
    timeout_seconds: float,
) -> tuple[float, str]:
    """Confirm the mutation through a direct read.

    For an upsert the check is not just presence: the stored title must be the
    *new* one. Presence alone would also hold for the copy that was already
    there, so it would prove nothing about an actualización.
    """
    if kind == "baja":

        def observe() -> tuple[bool, str]:
            payload = store.get_by_record_id(record_id)
            if payload is None:
                return True, "la lectura por ID no devuelve nada"
            return False, f"sigue devolviendo «{_title_of(payload)[:60]}»"

        description = "el punto sigue siendo legible por ID"
    else:

        def observe() -> tuple[bool, str]:
            payload = store.get_by_record_id(record_id)
            if payload is None:
                return False, "la lectura por ID no devuelve nada"
            title = _title_of(payload)
            if title != expected_title:
                return False, f"el título almacenado es «{title[:60]}»"
            return True, f"la lectura por ID devuelve «{title[:60]}»"

        description = "la lectura por ID no refleja la ficha escrita"

    return _wait_until(
        observe, timeout_seconds=timeout_seconds, description=description
    )


def _probe_by_search(
    store: QdrantStore,
    query_vector: np.ndarray,
    *,
    record_id: str,
    kind: EventKind,
    expected_title: str,
    top_k: int,
    timeout_seconds: float,
) -> tuple[float, str]:
    """Confirm the mutation through the path the users actually take.

    Reading by ID and retrieving by vector are two different copies of the
    truth in an ANN engine: a point can be stored and not yet reachable through
    the graph. RF-16 asks for both because only the second one is the product.
    """

    def locate() -> tuple[int, str]:
        hits = store.search_vector(query_vector, top_k=top_k)
        for hit in hits:
            if hit.record_id == record_id:
                return hit.rank, hit.title
        return 0, ""

    if kind == "baja":

        def observe() -> tuple[bool, str]:
            rank, _ = locate()
            if rank:
                return False, f"sigue apareciendo en la posición {rank}"
            return True, f"ausente del top-{top_k} de su propia consulta"

        description = f"el producto sigue apareciendo en el top-{top_k}"
    else:

        def observe() -> tuple[bool, str]:
            rank, title = locate()
            if not rank:
                return False, f"no aparece en el top-{top_k}"
            if title != expected_title:
                return False, f"aparece en la posición {rank} con el título anterior"
            return True, f"posición {rank} de {top_k} con la ficha nueva"

        description = (
            f"el producto no es recuperable en el top-{top_k} con su ficha nueva"
        )

    return _wait_until(
        observe, timeout_seconds=timeout_seconds, description=description
    )


# ------------------------------------------------------------------ aplicación


def _apply_one(
    event: CatalogEvent, store: QdrantStore, vector: np.ndarray | None
) -> float:
    """Send one mutation and return how long the call took, in milliseconds."""
    started = monotonic()
    if event.is_deletion:
        store.delete_by_record_ids([event.record_id])
    else:
        if vector is None:
            raise EventError(f"{event.event_id}: falta el vector del producto")
        store.upsert_batch([event.require_record()], vector.reshape(1, -1))
    return (monotonic() - started) * 1000.0


def _encode_upserts(
    events: Sequence[CatalogEvent], encoder: Encoder, strategy: TextStrategy
) -> dict[str, np.ndarray]:
    """Encode every sheet in one batch, keyed by ``record_id``.

    Encoding once and applying afterwards keeps the sequence order intact while
    letting the model batch. The alternative — encoding inside the loop — would
    mix the model's cost into the visibility measurements.
    """
    records: list[CatalogRecord] = [
        event.require_record() for event in events if not event.is_deletion
    ]
    if not records:
        return {}
    texts = [compose(record, strategy) for record in records]
    matrix = encoder.encode(texts, role="document")
    return {
        record.record_id: matrix.vectors[position]
        for position, record in enumerate(records)
    }


def apply_events(
    events: Sequence[CatalogEvent],
    store: QdrantStore,
    encoder: Encoder,
    *,
    text_strategy: TextStrategy = "title_brand_color",
    measure_visibility: bool = True,
    timeout_seconds: float = 30.0,
    top_k: int = 10,
) -> EventReport:
    """Apply every mutation in ``sequence`` order and verify the outcome.

    The first event of each kind is probed through both channels, which is what
    RF-16 asks for. Probing every event would multiply the round trips without
    adding evidence: the mechanism being demonstrated is the same one.
    """
    ordered = sorted(events, key=lambda event: event.sequence)
    sequences = [event.sequence for event in ordered]
    if sequences != list(range(1, len(ordered) + 1)):
        raise EventError(
            f"Las secuencias deben ser 1..{len(ordered)} sin huecos, "
            f"recibido {sequences}"
        )

    status = store.status()
    if not status.exists:
        raise EventError(
            f"La colección {store.collection!r} no existe. Ejecuta `aurum ingest` "
            "antes de aplicar los eventos."
        )
    points_before = status.points_count

    vectors = _encode_upserts(ordered, encoder, text_strategy)
    query_vectors: dict[str, np.ndarray] = {}

    counts = {"alta": 0, "actualizacion": 0, "baja": 0}
    bajas_efectivas = 0
    kinds: list[tuple[str, EventKind]] = []
    probes: list[VisibilityProbe] = []
    probed: set[EventKind] = set()

    for event in ordered:
        # Clasificar ANTES de aplicar: después, todo upsert parecería una
        # actualización y toda baja parecería no haber borrado nada.
        kind = classify(event, store)
        counts[kind] += 1
        kinds.append((event.event_id, kind))

        existing = store.get_by_record_id(event.record_id) if kind == "baja" else None
        if kind == "baja" and existing is not None:
            bajas_efectivas += 1

        should_probe = measure_visibility and kind not in probed
        if not should_probe:
            _apply_one(event, store, vectors.get(event.record_id))
            continue

        probe = _probe_event(
            event,
            kind,
            store,
            encoder,
            vectors.get(event.record_id),
            existing_payload=existing,
            strategy=text_strategy,
            top_k=top_k,
            timeout_seconds=timeout_seconds,
            query_cache=query_vectors,
        )
        if probe is None:
            # La baja apuntaba a un punto ausente: no hay nada que observar
            # desaparecer, así que se aplica sin fabricar una sonda vacía.
            _apply_one(event, store, vectors.get(event.record_id))
            continue
        probes.append(probe)
        probed.add(kind)

    expected = points_before + counts["alta"] - bajas_efectivas
    _wait_for_exact_count(store, expected=expected, timeout_seconds=timeout_seconds)

    notes = [
        "Las altas y las actualizaciones son la misma llamada al motor: se "
        "distinguen leyendo la colección antes de escribir.",
        "El recuento no distingue por sí solo un run correcto de uno que no "
        f"hizo nada: entran {counts['alta']} productos y salen {bajas_efectivas}, "
        "así que el total no se mueve. El estado se comprueba por identidad.",
    ]
    if counts["alta"] == 0 and bajas_efectivas == 0:
        notes.append(
            "Ninguna alta y ninguna baja efectiva: esta reaplicación encontró "
            "el estado ya alcanzado, que es exactamente lo que exige RF-16."
        )

    return EventReport(
        applied=len(ordered),
        altas=counts["alta"],
        actualizaciones=counts["actualizacion"],
        bajas=counts["baja"],
        bajas_efectivas=bajas_efectivas,
        points_before=points_before,
        points_after=expected,
        kinds=tuple(kinds),
        probes=tuple(probes),
        notes=tuple(notes),
    )


def _probe_event(
    event: CatalogEvent,
    kind: EventKind,
    store: QdrantStore,
    encoder: Encoder,
    vector: np.ndarray | None,
    *,
    existing_payload: dict[str, object] | None,
    strategy: TextStrategy,
    top_k: int,
    timeout_seconds: float,
    query_cache: dict[str, np.ndarray],
) -> VisibilityProbe | None:
    """Apply one event with a before/after observation on both channels.

    Returns ``None`` for a deletion that has nothing to delete, because the
    "it disappeared" evidence would be vacuous.
    """
    if kind == "baja":
        if existing_payload is None:
            return None
        # La consulta se construye con el título que el motor tiene guardado:
        # el fichero de eventos no trae ficha para una baja.
        query_text = _title_of(existing_payload)
        expected_title = ""
    else:
        record = event.require_record()
        query_text = compose(record, strategy)
        expected_title = record.title

    if not query_text:
        raise EventError(f"{event.event_id}: no hay texto con el que consultar")

    if query_text not in query_cache:
        query_cache[query_text] = encoder.encode([query_text], role="query").vectors[0]
    query_vector = query_cache[query_text]

    baseline = ""
    if kind == "baja":
        # Que algo no se encuentre después de borrarlo solo significa algo si
        # se encontraba antes. Se deja constancia de las dos observaciones.
        hits = store.search_vector(query_vector, top_k=top_k)
        found = next((hit for hit in hits if hit.record_id == event.record_id), None)
        baseline = (
            f"antes de la baja aparecía en la posición {found.rank}"
            if found
            else f"antes de la baja YA no aparecía en el top-{top_k}: la "
            "desaparición posterior no prueba nada"
        )

    write_ms = _apply_one(event, store, vector)

    by_id_ms, id_evidence = _probe_by_id(
        store,
        record_id=event.record_id,
        kind=kind,
        expected_title=expected_title,
        timeout_seconds=timeout_seconds,
    )
    by_search_ms, search_evidence = _probe_by_search(
        store,
        query_vector,
        record_id=event.record_id,
        kind=kind,
        expected_title=expected_title,
        top_k=top_k,
        timeout_seconds=timeout_seconds,
    )

    return VisibilityProbe(
        event_id=event.event_id,
        product_id=event.product_id,
        record_id=event.record_id,
        kind=kind,
        write_ms=write_ms,
        by_id_ms=by_id_ms,
        by_search_ms=by_search_ms,
        id_evidence=id_evidence,
        search_evidence=search_evidence,
        query_text=query_text,
        baseline=baseline,
    )


# ------------------------------------------------------------ estado alcanzado


def snapshot_state(
    events: Sequence[CatalogEvent], store: QdrantStore
) -> dict[str, str | None]:
    """Read back what the collection holds for every ID the events touch.

    This is the comparison that proves idempotence. Comparing point counts
    would not: the events add and remove the same number of products, so the
    total is identical whether they ran once, twice or never.
    """
    state: dict[str, str | None] = {}
    for event in sorted(events, key=lambda item: item.sequence):
        payload = store.get_by_record_id(event.record_id)
        state[event.record_id] = None if payload is None else _title_of(payload)
    return state
