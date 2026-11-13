"""Markdown rendering of summaries, comparisons and flag listings."""

from __future__ import annotations

from collections.abc import Sequence

LABELS = {
    "coverage": "coverage",
    "density": "density",
    "compression": "compression",
    "novel_1": "novel unigrams",
    "novel_2": "novel bigrams",
    "novel_3": "novel trigrams",
    "novel_4": "novel 4-grams",
    "longest_span": "longest copied span (tokens)",
    "long_span_share": "share in spans >= k",
    "position_mean": "mean source position of copies",
    "lead_share": "share of copies from the lead",
    "order_concordance": "fragment pairs in source order",
}


def _f(v: object, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _ci(ci: Sequence[float] | None) -> str:
    return "-" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _pct(share: float | None, ci: Sequence[float] | None) -> str:
    if share is None:
        return "-"
    s = f"{100 * share:.1f} %"
    if ci is not None:
        s += f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"
    return s


def _header(meta: dict) -> list[str]:
    o = meta["options"]
    lines = [
        f"Records: {meta['records']}.",
        f"Tokeniser: {meta['tokenizer']}.",
        f"Long spans: at least {o['span_k']} tokens; position and order use fragments of at least "
        f"{o['min_span']} tokens; lead = first {100 * o['lead']:g} % of the source.",
        f"Fragments: {o.get('algorithm', 'exact')} search.",
    ]
    level = f"{100 * o['level']:g} %"
    if meta.get("group_field"):
        lines.append(
            f"Intervals: {level} percentile bootstrap, {o['n_boot']} resamples, seed {o['seed']}, "
            f"clustered by `{meta['group_field']}` ({meta['groups']} groups)."
        )
    else:
        lines.append(
            f"Intervals: {level} percentile bootstrap over records, {o['n_boot']} resamples, "
            f"seed {o['seed']}."
        )
    return lines


def summary_markdown(meta: dict) -> str:
    out = ["# copyspan summary", ""]
    out += _header(meta)
    out += [
        "",
        "| measure | n | mean | interval | p10 | median | p90 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in meta["metrics"].items():
        q = m["quantiles"]
        out.append(
            f"| {LABELS.get(name, name)} | {m['n']} | {_f(m['mean'])} | {_ci(m['ci'])} | "
            f"{_f(q.get('q10'))} | {_f(q.get('q50'))} | {_f(q.get('q90'))} |"
        )
    t = meta["tails"]
    out += ["", f"## Tails (of {t['records_with_output']} records with a non-empty output)", ""]
    out += ["| tail | count | share [interval] |", "|---|---|---|"]
    for row in t["fully_copied"]:
        th = f"{row['threshold']:g}"
        c = row["coverage"]
        out.append(f"| coverage >= {th} | {c['count']} | {_pct(c['share'], c['ci'])} |")
        c = row["long_span_share"]
        out.append(
            f"| share in spans >= k at least {th} | {c['count']} | {_pct(c['share'], c['ci'])} |"
        )
    for key, label in (
        ("whole_output_one_span", "whole output is one source span"),
        ("no_token_copied", "no token copied"),
        ("no_shared_content_word", "no shared content word"),
    ):
        c = t[key]
        out.append(f"| {label} | {c['count']} | {_pct(c['share'], c['ci'])} |")
    hist = meta["position_histogram"]
    if any(hist):
        out += ["", "## Where copies come from", ""]
        out.append(
            "Share of copied tokens by tenth of the source (fragments of at least "
            f"{meta['options']['min_span']} tokens):"
        )
        out += [
            "",
            "| " + " | ".join(f"{10 * k}-{10 * (k + 1)} %" for k in range(len(hist))) + " |",
            "|" + "---|" * len(hist),
            "| " + " | ".join(f"{100 * h:.1f}" for h in hist) + " |",
        ]
    ok = meta["order_kept"]
    if ok["records"]:
        out += [
            "",
            f"Order kept (all fragments in source order) in {_pct(ok['share'], None)} of "
            f"{ok['records']} records with two or more such fragments.",
        ]
    if meta["flags"]:
        out += ["", "Flags: " + ", ".join(f"{k} {v}" for k, v in meta["flags"].items()) + "."]
    return "\n".join(out) + "\n"


def compare_markdown(meta: dict) -> str:
    out = ["# copyspan comparison", ""]
    out += _header(meta)
    out.append(
        f"Output a: `{meta['label_a']}`; output b: `{meta['label_b']}`. Differences are a - b."
    )
    if meta.get("join"):
        j = meta["join"]
        out.append(
            f"Joined by id: {j['matched']} matched, {j['only_a']} only in a, "
            f"{j['only_b']} only in b."
        )
    out += [
        "",
        "| measure | pairs | mean a | mean b | a - b | interval |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in meta["metrics"].items():
        out.append(
            f"| {LABELS.get(name, name)} | {m['pairs']} | {_f(m['mean_a'])} | {_f(m['mean_b'])} | "
            f"{_f(m['diff'])} | {_ci(m['ci'])} |"
        )
    out += ["", "| tail | count a | count b | share a - b [interval] |", "|---|---|---|---|"]
    for name, t in meta["tails"].items():
        out.append(
            f"| {name.replace('_', ' ')} | {t['count_a']} | {t['count_b']} | "
            f"{_pct(t['diff'], t['ci'])} |"
        )
    return "\n".join(out) + "\n"


def flags_markdown(rows: list[dict], tokenizer: str) -> str:
    out = [
        "# copyspan flags",
        "",
        f"Tokeniser: {tokenizer}.",
        "",
        "| id | flags | coverage | longest span | n_output | n_source |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['id']} | {', '.join(r['flags'])} | {_f(r['coverage'])} | "
            f"{r['longest_span']} | {r['n_output']} | {r['n_source']} |"
        )
    if not rows:
        out.append("| (none) | | | | | |")
    return "\n".join(out) + "\n"
