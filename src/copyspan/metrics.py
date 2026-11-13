"""Per-record copy measures for one (source, output) pair of token lists."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from copyspan.fragments import Fragment, fragments, fragments_newsroom, longest_matches
from copyspan.stopwords import STOPWORDS

NGRAM_ORDERS = (1, 2, 3, 4)
ALGORITHMS = ("exact", "newsroom")


@dataclass(frozen=True)
class Options:
    """Settings that change per-record numbers.

    span_k: a token counts as "in a long span" when its fragment has at least
        this many tokens.
    min_span: fragments shorter than this are ignored for the position and
        order measures (single shared words are mostly function words).
    lead: the leading share of the source used for the lead measure.
    algorithm: "exact" (longest match at each output position) or "newsroom"
        (the published procedure with its source-index jump).
    """

    span_k: int = 3
    min_span: int = 3
    lead: float = 0.25
    algorithm: str = "exact"

    def __post_init__(self) -> None:
        if self.span_k < 1 or self.min_span < 1:
            raise ValueError("span_k and min_span must be at least 1")
        if not 0.0 < self.lead <= 1.0:
            raise ValueError("lead must be in (0, 1]")
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}")


@dataclass
class RecordScore:
    """All measures for one record. ``None`` marks a measure that is undefined."""

    n_source: int
    n_output: int
    coverage: float | None
    density: float | None
    compression: float | None
    novel: dict[int, float | None]
    longest_span: int
    long_span_share: float | None
    n_fragments: int
    position_mean: float | None
    lead_share: float | None
    order_kept: bool | None
    order_concordance: float | None
    flags: list[str] = field(default_factory=list)
    spans: list[Fragment] = field(default_factory=list)

    def to_dict(self, with_spans: bool = False) -> dict:
        d = {
            "n_source": self.n_source,
            "n_output": self.n_output,
            "coverage": self.coverage,
            "density": self.density,
            "compression": self.compression,
        }
        for n in sorted(self.novel):
            d[f"novel_{n}"] = self.novel[n]
        d.update(
            {
                "longest_span": self.longest_span,
                "long_span_share": self.long_span_share,
                "n_fragments": self.n_fragments,
                "position_mean": self.position_mean,
                "lead_share": self.lead_share,
                "order_kept": self.order_kept,
                "order_concordance": self.order_concordance,
                "flags": list(self.flags),
            }
        )
        if with_spans:
            d["spans"] = [f.to_dict() for f in self.spans]
        return d


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def novel_ngram_share(source: Sequence[str], output: Sequence[str], n: int) -> float | None:
    """Share of the output's n-grams (counted with repetition) absent from the source.

    ``None`` when the output has fewer than ``n`` tokens.
    """
    out = _ngrams(output, n)
    if not out:
        return None
    src = set(_ngrams(source, n))
    return sum(1 for g in out if g not in src) / len(out)


def is_content_word(token: str) -> bool:
    return any(ch.isalnum() for ch in token) and token.casefold() not in STOPWORDS


def score_tokens(
    source: Sequence[str], output: Sequence[str], options: Options | None = None
) -> RecordScore:
    """Score one pair of token lists."""
    opt = options or Options()
    n_src, n_out = len(source), len(output)
    lengths, starts = longest_matches(source, output)
    finder = fragments if opt.algorithm == "exact" else fragments_newsroom
    frags = finder(source, output) if n_out and n_src else []
    copied = sum(f.length for f in frags)

    if n_out:
        coverage: float | None = copied / n_out
        density: float | None = sum(f.length**2 for f in frags) / n_out
        compression: float | None = n_src / n_out
        long_share: float | None = sum(f.length for f in frags if f.length >= opt.span_k) / n_out
    else:
        coverage = density = compression = long_share = None

    novel = {n: novel_ngram_share(source, output, n) for n in NGRAM_ORDERS}
    longest = max(lengths) if lengths else 0

    # Where the copies come from, using fragments of at least min_span tokens.
    big = [f for f in frags if f.length >= opt.min_span]
    position_mean = lead_share = None
    if big and n_src:
        total = sum(f.length for f in big)
        # Mean relative position of the copied tokens: token at source index p
        # sits at (p + 0.5) / n_src.
        pos_sum = sum(f.length * (f.src_start + f.length / 2) for f in big)
        position_mean = pos_sum / (total * n_src)
        cut = opt.lead * n_src
        in_lead = sum(max(0.0, min(f.src_start + f.length, cut) - f.src_start) for f in big)
        lead_share = in_lead / total
    order_kept = concordance = None
    if len(big) >= 2:
        s = [f.src_start for f in big]
        order_kept = all(s[i] <= s[i + 1] for i in range(len(s) - 1))
        pairs = len(s) * (len(s) - 1) // 2
        conc = sum(1 for i in range(len(s)) for j in range(i + 1, len(s)) if s[i] <= s[j])
        concordance = conc / pairs

    flags: list[str] = []
    if n_out == 0:
        flags.append("empty_output")
    else:
        if n_src and longest == n_out:
            flags.append("prefix" if list(source[:n_out]) == list(output) else "slice")
        out_content = {t for t in output if is_content_word(t)}
        src_content = {t for t in source if is_content_word(t)}
        if not out_content & src_content:
            flags.append("no_shared_content")
        if n_out > n_src:
            flags.append("longer_than_source")

    return RecordScore(
        n_source=n_src,
        n_output=n_out,
        coverage=coverage,
        density=density,
        compression=compression,
        novel=novel,
        longest_span=longest,
        long_span_share=long_share,
        n_fragments=len(frags),
        position_mean=position_mean,
        lead_share=lead_share,
        order_kept=order_kept,
        order_concordance=concordance,
        flags=flags,
        spans=frags,
    )
