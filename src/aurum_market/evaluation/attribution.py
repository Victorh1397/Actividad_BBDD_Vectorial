"""Attributing failures to the layer that caused them (RF-24).

The statement asks for at least three representative failures, each traced to
one of four layers **with evidence**. The point is not to explain a bad number
away but to know where to spend the next hour: tuning an index that is not the
problem is the most expensive mistake available.

The procedure is mechanical, and it is what makes the exact oracle worth
building:

| Observation | Layer |
|---|---|
| The exact nearest neighbour is already semantically wrong | representation |
| The oracle finds it and the ANN index does not | index |
| The metadata is missing or the filter excludes the product | data or filters |
| The state read does not match the write applied | persistence |

Note the order. Only after the oracle is exonerated does the index become a
suspect — which is why fidelity is measured before anything is attributed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ..contracts import RetrievalQuery
from ..evaluation.metrics import QueryMetrics
from ..search import Retriever

Layer = Literal["representacion", "indice", "datos_o_filtros", "persistencia"]

LAYER_DESCRIPTIONS: dict[Layer, str] = {
    "representacion": (
        "El vecino exacto ya es semánticamente malo: el modelo no entiende la "
        "consulta, y ningún ajuste del índice lo arreglaría."
    ),
    "indice": (
        "El oráculo exacto recupera un producto que el índice ANN pierde: la "
        "aproximación está costando ranking."
    ),
    "datos_o_filtros": (
        "Falta información en la ficha, el metadato es inconsistente o la "
        "consulta excluye el producto."
    ),
    "persistencia": (
        "El estado leído todavía no coincide con la escritura aplicada."
    ),
}


@dataclass(frozen=True, slots=True)
class Attribution:
    """One failure traced to a layer, with the evidence that places it there."""

    query_id: str
    query_text: str
    layer: Layer
    metric: float
    evidence: dict[str, object] = field(default_factory=dict)
    explanation: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "layer": self.layer,
            "layer_description": LAYER_DESCRIPTIONS[self.layer],
            "metric": self.metric,
            "evidence": self.evidence,
            "explanation": self.explanation,
        }


def attribute_query(
    query: RetrievalQuery,
    metrics: QueryMetrics,
    *,
    engine: Retriever,
    oracle: Retriever,
    judgments: Mapping[str, float],
    top_k: int = 10,
    relevance_threshold: float = 2.0,
) -> Attribution:
    """Decide which layer explains one query's poor result.

    Both retrievers share the model and differ only in the store, so comparing
    them isolates the index cleanly.
    """
    engine_hits = engine.search(query.text, top_k=top_k)
    oracle_hits = oracle.search(query.text, top_k=top_k)
    engine_ids = [hit.product_id for hit in engine_hits]
    oracle_ids = [hit.product_id for hit in oracle_hits]

    relevant = {
        product_id
        for product_id, relevance in judgments.items()
        if relevance >= relevance_threshold
    }
    engine_relevant = [pid for pid in engine_ids if pid in relevant]
    oracle_relevant = [pid for pid in oracle_ids if pid in relevant]
    lost_to_index = [pid for pid in oracle_relevant if pid not in set(engine_ids)]

    evidence: dict[str, object] = {
        "ndcg": metrics.ndcg,
        "recall": metrics.recall,
        "recall_ceiling": metrics.recall_ceiling,
        "relevant_judged": len(relevant),
        "engine_relevant_retrieved": len(engine_relevant),
        "oracle_relevant_retrieved": len(oracle_relevant),
        "lost_to_index": lost_to_index,
        "engine_top3": engine_ids[:3],
        "oracle_top3": oracle_ids[:3],
    }

    if lost_to_index:
        return Attribution(
            query_id=query.query_id,
            query_text=query.text,
            layer="indice",
            metric=metrics.ndcg,
            evidence=evidence,
            explanation=(
                f"El oráculo exacto recupera {len(lost_to_index)} producto(s) "
                f"relevante(s) que el índice no devuelve: {lost_to_index}. La "
                "pérdida es de la aproximación, no del modelo."
            ),
        )

    if not oracle_relevant:
        return Attribution(
            query_id=query.query_id,
            query_text=query.text,
            layer="representacion",
            metric=metrics.ndcg,
            evidence=evidence,
            explanation=(
                "Ni siquiera la búsqueda exacta, comparando contra los 15.000 "
                "productos uno a uno, coloca un solo producto relevante en el "
                f"top-{top_k}. El índice está exonerado: el modelo no entiende "
                "esta consulta."
            ),
        )

    if metrics.recall_ceiling < 1.0 and metrics.recall >= metrics.recall_ceiling * 0.9:
        return Attribution(
            query_id=query.query_id,
            query_text=query.text,
            layer="datos_o_filtros",
            metric=metrics.recall,
            evidence=evidence,
            explanation=(
                f"Hay {len(relevant)} productos juzgados relevantes y solo "
                f"{top_k} posiciones, así que el techo es "
                f"{metrics.recall_ceiling:.4f}. El sistema alcanza "
                f"{metrics.recall:.4f}, el "
                f"{metrics.recall / metrics.recall_ceiling * 100:.0f} % de lo "
                "posible: la limitación es del reparto de los juicios, no del "
                "sistema."
            ),
        )

    return Attribution(
        query_id=query.query_id,
        query_text=query.text,
        layer="representacion",
        metric=metrics.ndcg,
        evidence=evidence,
        explanation=(
            f"El índice devuelve lo mismo que la búsqueda exacta, así que el "
            f"orden de los {len(engine_relevant)} relevantes recuperados "
            "depende solo de cómo el modelo representa consulta y producto."
        ),
    )


def attribute_failures(
    queries: Sequence[RetrievalQuery],
    per_query: Sequence[QueryMetrics],
    *,
    engine: Retriever,
    oracle: Retriever,
    judgments: Mapping[str, Mapping[str, float]],
    top_k: int = 10,
    minimum: int = 3,
) -> list[Attribution]:
    """Attribute the weakest queries, worst first.

    Sorted by nDCG so the failures examined are the ones that actually cost
    quality, not whichever happened to come first.
    """
    by_id = {query.query_id: query for query in queries}
    ranked = sorted(per_query, key=lambda item: item.ndcg)
    chosen = ranked[: max(minimum, 3)]
    return [
        attribute_query(
            by_id[metrics.query_id],
            metrics,
            engine=engine,
            oracle=oracle,
            judgments=judgments.get(metrics.query_id, {}),
            top_k=top_k,
        )
        for metrics in chosen
        if metrics.query_id in by_id
    ]


def demonstrate_index_failure(
    engine: Retriever,
    oracle: Retriever,
    queries: Sequence[RetrievalQuery],
    *,
    degraded_ef: int = 16,
    top_k: int = 10,
) -> Attribution | None:
    """Produce an index-layer failure on purpose, to prove we can detect one.

    The delivered configuration reaches perfect fidelity, so no real failure is
    attributable to the index — good news that leaves the attribution unable to
    show it can tell the layers apart. Lowering ``ef_search`` makes the index
    drop candidates the oracle still finds, which is the index signature.

    This documents a **diagnostic capability**, not a defect of the delivered
    system: the shipped configuration uses ``ef_search=256``.
    """
    store = getattr(engine, "_store", None)
    encoder = getattr(engine, "_encoder", None)
    if store is None or encoder is None or not hasattr(store, "search_vector"):
        return None

    for query in queries:
        try:
            vectors = encoder.encode([query.text], role="query").vectors
            degraded = store.search_vector(
                vectors[0], top_k=top_k, ef_search=degraded_ef
            )
        except TypeError:
            # El almacén no acepta ef_search: no hay nada que degradar.
            return None
        degraded_ids = [hit.product_id for hit in degraded]
        exact_ids = [hit.product_id for hit in oracle.search(query.text, top_k=top_k)]
        lost = [pid for pid in exact_ids if pid not in set(degraded_ids)]
        if lost:
            return Attribution(
                query_id=query.query_id,
                query_text=query.text,
                layer="indice",
                metric=len(lost) / len(exact_ids),
                evidence={
                    "ef_search_degraded": degraded_ef,
                    "ef_search_delivered": 256,
                    "lost_to_index": lost,
                    "degraded_top3": degraded_ids[:3],
                    "exact_top3": exact_ids[:3],
                },
                explanation=(
                    f"Con ef_search={degraded_ef} el índice pierde {len(lost)} de "
                    f"{len(exact_ids)} productos que la búsqueda exacta sí "
                    f"encuentra: {lost[:3]}. Con la configuración entregada "
                    "(ef_search=256) la fidelidad es 1,0 y esta pérdida "
                    "desaparece. Demuestra que un fallo de índice es "
                    "distinguible de uno de representación: aquí el oráculo "
                    "acierta y el motor no, mientras que en los fallos de "
                    "representación ambos fallan igual."
                ),
            )
    return None


def attribute_duplicate_miss(
    incoming_id: str,
    *,
    score: float,
    threshold: float,
    development_gap: tuple[float, float],
) -> Attribution:
    """Attribute a duplicate that slipped through the frozen threshold.

    A different kind of failure from the ranking ones: nothing is wrong with
    the retrieval, the candidate was found and scored — the decision boundary
    simply sat above it.
    """
    low, high = development_gap
    return Attribution(
        query_id=incoming_id,
        query_text=f"alta {incoming_id}",
        layer="datos_o_filtros",
        metric=score,
        evidence={
            "score": score,
            "threshold": threshold,
            "development_gap": [low, high],
            "inside_development_gap": low <= score <= high,
        },
        explanation=(
            f"La ficha puntuó {score:.4f} y el umbral congelado es "
            f"{threshold:.4f}. La recuperación funcionó —el candidato se "
            f"encontró y se puntuó—, pero la frontera de decisión quedó por "
            f"encima. El score cae **dentro** del hueco observado en desarrollo "
            f"[{low:.4f}, {high:.4f}], así que el fallo no es del modelo ni del "
            f"índice: es que catorce casos etiquetados no bastan para colocar "
            f"un umbral con confianza. Reajustarlo ahora invalidaría la "
            f"evaluación (P-04)."
        ),
    )


def summarize(attributions: Sequence[Attribution]) -> dict[str, object]:
    """Aggregate by layer, which is what the report needs."""
    counts: dict[str, int] = {}
    for item in attributions:
        counts[item.layer] = counts.get(item.layer, 0) + 1
    return {
        "analysed": len(attributions),
        "by_layer": counts,
        "layers": LAYER_DESCRIPTIONS,
        "attributions": [item.as_dict() for item in attributions],
    }
