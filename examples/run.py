"""Score the example pairs, summarise them and compare two outputs, using the Python API.

Run from the repository root:  python examples/run.py
"""

from __future__ import annotations

import json
from pathlib import Path

from copyspan import Tokenizer
from copyspan.io import read_records
from copyspan.metrics import Options, score_tokens
from copyspan.summary import BootstrapSettings, compare, summarize

HERE = Path(__file__).parent


def main() -> None:
    tok = Tokenizer("word", lower=True)
    opt = Options(span_k=3)
    recs = read_records(HERE / "pairs.jsonl", "document", "summary", "id", "topic")
    drafts = read_records(HERE / "pairs.jsonl", "document", "draft", "id", "topic")

    summaries = [score_tokens(tok(r.source), tok(r.output), opt) for r in recs]
    for rec, s in zip(recs, summaries, strict=True):
        spans = [" ".join(tok(rec.output)[f.out_start : f.out_start + f.length]) for f in s.spans]
        long_spans = [x for x in spans if len(x.split()) >= opt.span_k]
        print(f"{rec.id}: coverage {s.coverage:.2f}, density {s.density:.2f}, flags {s.flags}")
        print(f"    copied spans of 3+ tokens: {long_spans}")

    groups = [r.group for r in recs]
    boot = BootstrapSettings(n_boot=1000, seed=1)
    summary = summarize(summaries, groups, opt, boot=boot)
    cov = summary["metrics"]["coverage"]
    lo, hi = cov["ci"]
    print(f"\nmean coverage {cov['mean']:.3f}, interval clustered by topic [{lo:.3f}, {hi:.3f}]")

    draft_scores = [score_tokens(tok(r.source), tok(r.output), opt) for r in drafts]
    diff = compare(summaries, draft_scores, groups, boot=boot)["metrics"]["density"]
    lo, hi = diff["ci"]
    print(f"density, summary minus draft: {diff['diff']:.3f} [{lo:.3f}, {hi:.3f}]")
    print("no shared content word:", json.dumps(summary["tails"]["no_shared_content_word"]))


if __name__ == "__main__":
    main()
