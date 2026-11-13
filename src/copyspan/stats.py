"""Quantiles and bootstrap intervals over records, optionally clustered by group."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
_CHUNK_CELLS = 2_000_000  # resample indices held in memory at once


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def to_list(self) -> list[float]:
        return [self.low, self.high]


def quantiles(values: Sequence[float], qs: Sequence[float] = QUANTILES) -> dict[str, float]:
    """Linear-interpolation quantiles (numpy's default, type 7)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {}
    return {f"q{round(q * 100):d}": float(v) for q, v in zip(qs, np.quantile(arr, qs), strict=True)}


def _cluster_totals(
    values: np.ndarray, groups: Sequence[str] | None
) -> tuple[np.ndarray, np.ndarray]:
    if groups is None:
        return values, np.ones_like(values)
    _, inverse = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(float)
    return sums, counts


def bootstrap_mean(
    values: Sequence[float],
    groups: Sequence[str] | None = None,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval | None:
    """Percentile bootstrap interval for the mean over records.

    With ``groups``, whole groups are resampled with replacement and the mean is
    the ratio of the resampled sums to the resampled record counts, so records
    of one group move together. Returns ``None`` for no values.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    if groups is not None and len(groups) != arr.size:
        raise ValueError("groups and values differ in length")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    sums, counts = _cluster_totals(arr, groups)
    g = sums.size
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    step = max(1, _CHUNK_CELLS // g)
    for start in range(0, n_boot, step):
        stop = min(n_boot, start + step)
        idx = rng.integers(0, g, size=(stop - start, g))
        stats[start:stop] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(stats, [alpha, 1.0 - alpha])
    return Interval(float(arr.mean()), float(low), float(high))


def paired_difference(
    a: Sequence[float],
    b: Sequence[float],
    groups: Sequence[str] | None = None,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval | None:
    """Interval for mean(a - b) over paired records, resampling pairs (or groups) together."""
    if len(a) != len(b):
        raise ValueError("paired samples differ in length")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_mean(diff, groups, n_boot=n_boot, level=level, seed=seed)
