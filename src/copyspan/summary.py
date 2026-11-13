"""Dataset summaries and paired comparisons built from per-record scores."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from copyspan.metrics import NGRAM_ORDERS, Options, RecordScore
from copyspan.stats import bootstrap_mean, paired_difference, quantiles

METRICS = (
    "coverage",
    "density",
    "compression",
    *(f"novel_{n}" for n in NGRAM_ORDERS),
    "longest_span",
    "long_span_share",
    "position_mean",
    "lead_share",
    "order_concordance",
)
POSITION_BINS = 10


@dataclass(frozen=True)
class BootstrapSettings:
    n_boot: int = 2000
    level: float = 0.95
    seed: int = 0


def metric_value(score: RecordScore, name: str) -> float | None:
    if name.startswith("novel_"):
        return score.novel.get(int(name.split("_")[1]))
    v = getattr(score, name)
    return None if v is None else float(v)


def _defined(
    scores: Sequence[RecordScore], name: str, groups: Sequence[str] | None
) -> tuple[list[float], list[str] | None]:
    vals: list[float] = []
    grp: list[str] | None = [] if groups is not None else None
    for k, s in enumerate(scores):
        v = metric_value(s, name)
        if v is None:
            continue
        vals.append(v)
        if grp is not None and groups is not None:
            grp.append(groups[k])
    return vals, grp


def _share(hits: Sequence[bool], groups: Sequence[str] | None, boot: BootstrapSettings) -> dict:
    iv = bootstrap_mean(
        [1.0 if h else 0.0 for h in hits], groups, boot.n_boot, boot.level, boot.seed
    )
    return {
        "count": int(sum(hits)),
        "share": None if iv is None else iv.estimate,
        "ci": None if iv is None else iv.to_list(),
    }


def position_histogram(
    scores: Sequence[RecordScore], options: Options, bins: int = POSITION_BINS
) -> list[float]:
    """Share of copied tokens (fragments of at least ``min_span``) per tenth of the source.

    Every copied token is placed by its relative position in its source; the
    histogram pools tokens over records.
    """
    hist = np.zeros(bins)
    for s in scores:
        if not s.n_source:
            continue
        for f in s.spans:
            if f.length < options.min_span:
                continue
            pos = (np.arange(f.src_start, f.src_start + f.length) + 0.5) / s.n_source
            hist += np.bincount(np.minimum((pos * bins).astype(int), bins - 1), minlength=bins)
    total = hist.sum()
    return [float(x) for x in (hist / total if total else hist)]


def summarize(
    scores: Sequence[RecordScore],
    groups: Sequence[str] | None = None,
    options: Options | None = None,
    full_thresholds: Sequence[float] = (0.9, 1.0),
    boot: BootstrapSettings | None = None,
) -> dict:
    """Distributions, intervals and tails over a dataset."""
    opt = options or Options()
    bs = boot or BootstrapSettings()
    if groups is not None and len(groups) != len(scores):
        raise ValueError("groups and scores differ in length")
    metrics = {}
    for name in METRICS:
        vals, grp = _defined(scores, name, groups)
        iv = bootstrap_mean(vals, grp, bs.n_boot, bs.level, bs.seed)
        metrics[name] = {
            "n": len(vals),
            "mean": None if iv is None else iv.estimate,
            "ci": None if iv is None else iv.to_list(),
            "quantiles": quantiles(vals),
        }

    nonempty = [k for k, s in enumerate(scores) if s.n_output]
    ne_scores = [scores[k] for k in nonempty]
    ne_groups = [groups[k] for k in nonempty] if groups is not None else None
    full = []
    for t in sorted(full_thresholds):
        full.append(
            {
                "threshold": t,
                "coverage": _share([s.coverage >= t for s in ne_scores], ne_groups, bs),
                "long_span_share": _share(
                    [s.long_span_share >= t for s in ne_scores], ne_groups, bs
                ),
            }
        )
    tails = {
        "records_with_output": len(ne_scores),
        "fully_copied": full,
        "whole_output_one_span": _share(
            [("prefix" in s.flags or "slice" in s.flags) for s in ne_scores], ne_groups, bs
        ),
        "no_token_copied": _share([s.coverage == 0 for s in ne_scores], ne_groups, bs),
        "no_shared_content_word": _share(
            ["no_shared_content" in s.flags for s in ne_scores], ne_groups, bs
        ),
    }
    flag_counts = Counter(f for s in scores for f in s.flags)
    ordered = [s.order_kept for s in scores if s.order_kept is not None]
    return {
        "records": len(scores),
        "groups": len(set(groups)) if groups is not None else None,
        "options": {
            "span_k": opt.span_k,
            "min_span": opt.min_span,
            "lead": opt.lead,
            "algorithm": opt.algorithm,
            "n_boot": bs.n_boot,
            "level": bs.level,
            "seed": bs.seed,
        },
        "metrics": metrics,
        "tails": tails,
        "flags": dict(sorted(flag_counts.items())),
        "position_histogram": position_histogram(scores, opt),
        "order_kept": {
            "records": len(ordered),
            "share": (sum(ordered) / len(ordered)) if ordered else None,
        },
    }


def compare(
    scores_a: Sequence[RecordScore],
    scores_b: Sequence[RecordScore],
    groups: Sequence[str] | None = None,
    full_thresholds: Sequence[float] = (0.9, 1.0),
    boot: BootstrapSettings | None = None,
) -> dict:
    """Paired comparison of two outputs for the same sources (``a - b``)."""
    bs = boot or BootstrapSettings()
    if len(scores_a) != len(scores_b):
        raise ValueError("the two score lists differ in length")
    if groups is not None and len(groups) != len(scores_a):
        raise ValueError("groups and scores differ in length")
    out: dict = {"records": len(scores_a), "metrics": {}, "tails": {}}
    for name in METRICS:
        a_vals, b_vals, grp = [], [], []
        for k, (sa, sb) in enumerate(zip(scores_a, scores_b, strict=True)):
            va, vb = metric_value(sa, name), metric_value(sb, name)
            if va is None or vb is None:
                continue
            a_vals.append(va)
            b_vals.append(vb)
            if groups is not None:
                grp.append(groups[k])
        iv = paired_difference(
            a_vals, b_vals, grp if groups is not None else None, bs.n_boot, bs.level, bs.seed
        )
        out["metrics"][name] = {
            "pairs": len(a_vals),
            "mean_a": float(np.mean(a_vals)) if a_vals else None,
            "mean_b": float(np.mean(b_vals)) if b_vals else None,
            "diff": None if iv is None else iv.estimate,
            "ci": None if iv is None else iv.to_list(),
        }

    def tail_diff(pred) -> dict:
        ha = [1.0 if pred(s) else 0.0 for s in scores_a]
        hb = [1.0 if pred(s) else 0.0 for s in scores_b]
        iv = paired_difference(ha, hb, groups, bs.n_boot, bs.level, bs.seed)
        return {
            "count_a": int(sum(ha)),
            "count_b": int(sum(hb)),
            "diff": None if iv is None else iv.estimate,
            "ci": None if iv is None else iv.to_list(),
        }

    for t in sorted(full_thresholds):
        out["tails"][f"coverage>={t:g}"] = tail_diff(
            lambda s, t=t: s.coverage is not None and s.coverage >= t
        )
    out["tails"]["whole_output_one_span"] = tail_diff(
        lambda s: "prefix" in s.flags or "slice" in s.flags
    )
    out["tails"]["no_token_copied"] = tail_diff(lambda s: s.coverage == 0)
    out["tails"]["no_shared_content_word"] = tail_diff(lambda s: "no_shared_content" in s.flags)
    return out
