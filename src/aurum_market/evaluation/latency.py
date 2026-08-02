"""Query latency with declared warm-up and repetitions (RF-21).

The statement is explicit that this describes *this* execution and must not be
used to rank providers running on different infrastructure. So the numbers ship
with the environment that produced them.

Two things are separated on purpose. Encoding the query runs a neural network;
searching walks a graph. Reporting them together would hide which one dominates
— and measuring showed the intuition backwards: encoding takes ~2.6 ms while the
round-trip to Qdrant takes ~107 ms, so the search dominates by a factor of 40.

Adapted from ``measure_repeated_latency`` of session 01.
"""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Distribution of repeated single-query measurements, in milliseconds."""

    label: str
    samples_ms: tuple[float, ...]
    warmup_repetitions: int
    repetitions: int

    @property
    def count(self) -> int:
        return len(self.samples_ms)

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.samples_ms, 50))

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.samples_ms, 95))

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.samples_ms))

    @property
    def min_ms(self) -> float:
        return float(np.min(self.samples_ms))

    @property
    def max_ms(self) -> float:
        return float(np.max(self.samples_ms))

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "samples": self.count,
            "warmup_repetitions": self.warmup_repetitions,
            "repetitions": self.repetitions,
        }


def describe_environment(**extra: object) -> dict[str, object]:
    """Capture what the numbers depend on.

    A latency without its environment is not reproducible and invites exactly
    the cross-provider comparison the statement rules out.
    """
    return {
        "python_version": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "processor": platform.processor() or "desconocido",
        "cpu_count": _cpu_count(),
        "executable": sys.executable.split("\\")[-1],
        **extra,
    }


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 0


def measure_latency[T](
    operation: Callable[[T], object],
    inputs: Sequence[T],
    *,
    label: str,
    warmup_repetitions: int = 2,
    repetitions: int = 10,
) -> LatencySummary:
    """Time one operation repeatedly, discarding warm-up runs.

    Warm-up matters more than usual here: the first call loads model weights
    and Qdrant populates its caches, so including it would report a startup
    cost as if it were steady-state latency.

    Only a scalar duration is kept per call, so raising ``repetitions`` never
    retains result objects.
    """
    if not inputs:
        raise ValueError("Hace falta al menos una entrada que medir")
    if repetitions < 1:
        raise ValueError("repetitions debe ser >= 1")
    if warmup_repetitions < 0:
        raise ValueError("warmup_repetitions no puede ser negativo")

    for _ in range(warmup_repetitions):
        for value in inputs:
            operation(value)

    samples: list[float] = []
    for _ in range(repetitions):
        for value in inputs:
            started = perf_counter_ns()
            operation(value)
            samples.append((perf_counter_ns() - started) / 1_000_000.0)

    return LatencySummary(
        label=label,
        samples_ms=tuple(samples),
        warmup_repetitions=warmup_repetitions,
        repetitions=repetitions,
    )
