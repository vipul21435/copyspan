"""Command line interface: ``copyspan score | summary | compare | flag``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from copyspan import __version__
from copyspan.io import read_records
from copyspan.metrics import Options, score_tokens
from copyspan.tokenize import MODES, Tokenizer


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", help="JSON lines or CSV/TSV file; '-' reads JSON lines from stdin")
    p.add_argument("--source", required=True, help="name of the source text field")
    p.add_argument("--output", required=True, help="name of the output text field")
    p.add_argument("--id", dest="id_field", help="name of the id field (default: row number)")
    p.add_argument("--group", dest="group_field", help="group field for clustered intervals")
    p.add_argument("--tokenizer", choices=MODES, default="word", help="default: word")
    p.add_argument("--keep-case", action="store_true", help="do not fold case")
    p.add_argument("--span-k", type=int, default=3, help="long-span threshold in tokens (3)")
    p.add_argument("--min-span", type=int, default=3,
                   help="shortest fragment used for position and order measures (3)")
    p.add_argument("--lead", type=float, default=0.25,
                   help="leading share of the source for the lead measure (0.25)")


def _setup(args: argparse.Namespace) -> tuple[Tokenizer, Options]:
    tok = Tokenizer(args.tokenizer, lower=not args.keep_case)
    opt = Options(span_k=args.span_k, min_span=args.min_span, lead=args.lead)
    return tok, opt


def _fmt(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def cmd_score(args: argparse.Namespace) -> int:
    tok, opt = _setup(args)
    records = read_records(args.path, args.source, args.output, args.id_field, args.group_field)
    rows = []
    for rec in records:
        r = score_tokens(tok(rec.source), tok(rec.output), opt)
        d = {"id": rec.id}
        if rec.group is not None:
            d["group"] = rec.group
        d.update(r.to_dict(with_spans=args.spans))
        rows.append(d)
    out = sys.stdout
    if args.format == "jsonl":
        for d in rows:
            out.write(json.dumps(d) + "\n")
    elif args.format == "json":
        json.dump({"tokenizer": tok.describe(), "records": rows}, out, indent=2)
        out.write("\n")
    else:
        cols = ["id", "n_source", "n_output", "coverage", "density", "compression", "novel_2",
                "longest_span", "flags"]
        out.write(f"Tokeniser: {tok.describe()}.\n\n")
        out.write("| " + " | ".join(cols) + " |\n")
        out.write("|" + "---|" * len(cols) + "\n")
        for d in rows:
            cells = [", ".join(d[c]) if c == "flags" else _fmt(d[c]) for c in cols]
            out.write("| " + " | ".join(cells) + " |\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="copyspan", description="Measure how much of an output is copied from its source."
    )
    parser.add_argument("--version", action="version", version=f"copyspan {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("score", help="per-record measures")
    _common(p)
    p.add_argument("--format", choices=("jsonl", "json", "md"), default="jsonl")
    p.add_argument("--spans", action="store_true", help="include the copied spans")
    p.set_defaults(func=cmd_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as exc:
        print(f"copyspan: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
