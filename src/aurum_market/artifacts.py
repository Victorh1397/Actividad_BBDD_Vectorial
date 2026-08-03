"""Writing the delivery artifacts, validated against their contracts (RF-25).

Every artifact is checked against the JSON Schema that declares it **before**
it reaches disk. A file that does not meet its contract is never written, so
an invalid deliverable cannot exist — the checklist item is enforced rather
than reviewed.

The three files the statement asks for:

* ``resultados_busqueda.csv`` — 12 blind queries x 10 products
* ``resultados_duplicados.csv`` — 14 decisions with candidate and score
* ``metricas_desarrollo.json`` — nDCG@10, Recall@10, MRR@10 and latencies
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as SchemaValidationError

from .config import CONTRACTS_DIRECTORY, RESULTS_DIRECTORY
from .contracts import DuplicateDecision, SearchHit

SEARCH_RESULTS = "resultados_busqueda.csv"
DUPLICATE_RESULTS = "resultados_duplicados.csv"
DEVELOPMENT_METRICS = "metricas_desarrollo.json"


class ArtifactError(RuntimeError):
    """Raised when an artifact does not meet the contract that declares it."""


def _validate(payload: Any, contract: str) -> None:
    """Check a payload against its declared schema, or refuse to continue."""
    path = CONTRACTS_DIRECTORY / f"{contract}.schema.json"
    if not path.is_file():
        raise ArtifactError(f"Falta el contrato {path}")
    schema = json.loads(path.read_text(encoding="utf-8"))
    try:
        Draft202012Validator(schema).validate(payload)
    except SchemaValidationError as error:
        location = "/".join(str(part) for part in error.absolute_path) or "(raíz)"
        raise ArtifactError(
            f"{contract} incumple su contrato en {location}: {error.message}"
        ) from error


def build_search_rows(
    rankings: Mapping[str, Sequence[SearchHit]],
) -> list[dict[str, Any]]:
    """Turn per-query hits into the declared row shape.

    Ranks are re-derived from position instead of trusting the incoming hits:
    the contract demands 1..10 consecutive, and a store that returned an odd
    rank should not be able to leak it into the deliverable.
    """
    rows: list[dict[str, Any]] = []
    for query_id in sorted(rankings):
        hits = rankings[query_id]
        seen: set[str] = set()
        for position, hit in enumerate(hits, start=1):
            if hit.product_id in seen:
                raise ArtifactError(
                    f"{query_id} repite el producto {hit.product_id}: el contrato "
                    "exige diez IDs únicos por consulta"
                )
            seen.add(hit.product_id)
            rows.append(
                {
                    "evaluation_id": query_id,
                    "rank": position,
                    "product_id": hit.product_id,
                    # El score nativo llega hasta el fichero sin transformar
                    # (P-03): es una similitud coseno en [-1, 1].
                    "score": round(float(hit.native_score), 6),
                }
            )
    return rows


def write_search_results(
    rankings: Mapping[str, Sequence[SearchHit]],
    *,
    directory: Path | None = None,
) -> Path:
    """Write ``resultados_busqueda.csv`` after validating it."""
    rows = build_search_rows(rankings)
    _validate(rows, "resultados_busqueda")
    target = (directory or RESULTS_DIRECTORY) / SEARCH_RESULTS
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["evaluation_id", "rank", "product_id", "score"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return target


def build_duplicate_rows(
    decisions: Sequence[DuplicateDecision],
) -> list[dict[str, Any]]:
    """Turn decisions into the declared row shape."""
    return [
        {
            "incoming_id": decision.incoming_id,
            "predicted_duplicate": bool(decision.predicted_duplicate),
            "matched_product_id": decision.matched_product_id,
            "score": round(float(decision.score), 6),
        }
        for decision in sorted(decisions, key=lambda item: item.incoming_id)
    ]


def write_duplicate_results(
    decisions: Sequence[DuplicateDecision],
    *,
    directory: Path | None = None,
) -> Path:
    """Write ``resultados_duplicados.csv`` after validating it.

    The contract encodes the rule the checklist demands: a positive must name a
    product and a negative must not. Since ``DuplicateDecision`` already
    enforces it on construction, this is a second, independent gate.
    """
    rows = build_duplicate_rows(decisions)
    _validate(rows, "resultados_duplicados")
    target = (directory or RESULTS_DIRECTORY) / DUPLICATE_RESULTS
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "incoming_id",
                "predicted_duplicate",
                "matched_product_id",
                "score",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                # Los booleanos van en minúscula, como en el resto del dataset.
                {**row, "predicted_duplicate": str(row["predicted_duplicate"]).lower()}
            )
    return target


def write_development_metrics(
    payload: Mapping[str, Any], *, directory: Path | None = None
) -> Path:
    """Write ``metricas_desarrollo.json`` after validating it."""
    _validate(payload, "metricas_desarrollo")
    target = (directory or RESULTS_DIRECTORY) / DEVELOPMENT_METRICS
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return target


def read_search_results(*, directory: Path | None = None) -> list[dict[str, Any]]:
    """Read back the search artifact, coercing types for validation."""
    target = (directory or RESULTS_DIRECTORY) / SEARCH_RESULTS
    if not target.is_file():
        raise ArtifactError(f"No se encuentra {target}. Ejecuta `aurum deliver`.")
    with target.open(encoding="utf-8", newline="") as stream:
        return [
            {
                "evaluation_id": row["evaluation_id"],
                "rank": int(row["rank"]),
                "product_id": row["product_id"],
                "score": float(row["score"]),
            }
            for row in csv.DictReader(stream)
        ]


def read_duplicate_results(*, directory: Path | None = None) -> list[dict[str, Any]]:
    """Read back the duplicates artifact, coercing types for validation."""
    target = (directory or RESULTS_DIRECTORY) / DUPLICATE_RESULTS
    if not target.is_file():
        raise ArtifactError(f"No se encuentra {target}. Ejecuta `aurum deliver`.")
    with target.open(encoding="utf-8", newline="") as stream:
        return [
            {
                "incoming_id": row["incoming_id"],
                "predicted_duplicate": row["predicted_duplicate"].strip().lower()
                == "true",
                "matched_product_id": row["matched_product_id"],
                "score": float(row["score"]),
            }
            for row in csv.DictReader(stream)
        ]


def verify_artifacts(*, directory: Path | None = None) -> dict[str, str]:
    """Re-validate the three files on disk, as a final gate before delivery.

    Validating on write proves what we produced; validating on read proves what
    a corrector will actually open — the round trip through CSV could still
    lose or mangle something.
    """
    target = directory or RESULTS_DIRECTORY
    search = read_search_results(directory=target)
    duplicates = read_duplicate_results(directory=target)
    metrics_path = target / DEVELOPMENT_METRICS
    if not metrics_path.is_file():
        raise ArtifactError(f"No se encuentra {metrics_path}. Ejecuta `aurum deliver`.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    _validate(search, "resultados_busqueda")
    _validate(duplicates, "resultados_duplicados")
    _validate(metrics, "metricas_desarrollo")

    return {
        SEARCH_RESULTS: f"{len(search)} filas · "
        f"{len({row['evaluation_id'] for row in search})} consultas",
        DUPLICATE_RESULTS: f"{len(duplicates)} decisiones · "
        f"{sum(1 for row in duplicates if row['predicted_duplicate'])} positivas",
        DEVELOPMENT_METRICS: f"nDCG@10={metrics['ndcg_at_10']:.4f}",
    }
