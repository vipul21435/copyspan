"""Command line interface: ``copyspan score | summary | compare | flag``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from copyspan import __version__
from copyspan.io import Record, read_records
from copyspan.metrics import ALGORITHMS, Options, RecordScore, score_tokens
from copyspan.report import compare_markdown, flags_markdown, summary_markdown
from copyspan.summary import BootstrapSettings, compare, summarize
from copyspan.tokenize import MODES, Tokenizer

FLAG_NAMES = ("prefix", "slice", "no_shared_content", "longer_than_source", "empty_output")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("path", help="JSON lines or CSV/TSV file; '-' reads JSON lines from stdin")
    p.add_argument("--source", required=True, help="name of the source text field")
    p.add_argument("--output", required=True, help="name of the output text field")
    p.add_argument("--id", dest="id_field", help="name of the id field (default: row number)")
    p.add_argument("--group", dest="group_field", help="group field for clustered intervals")
    p.add_argument("--tokenizer", choices=MODES, default="word", help="default: word")
    p.add_argument("--keep-case", action="store_true", help="do not fold case")
    p.add_argument("--span-k", type=int, default=3, help="long-span threshold in tokens (3)")
    p.add_argument(
        "--min-span",
        type=int,
        default=3,
        help="shortest fragment used for position and order measures (3)",
    )
    p.add_argument(
        "--lead",
        type=float,
        default=0.25,
        help="leading share of the source for the lead measure (0.25)",
    )
    p.add_argument(
        "--algorithm",
        choices=ALGORITHMS,
        default="exact",
        help="fragment search: exact longest match (default) or the published newsroom procedure",
    )
    p.add_argument("--limit", type=int, help="use only the first N records")


def _boot_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--full",
        type=float,
        action="append",
        help="threshold for the fully-copied tail; repeatable (default 0.9 and 1.0)",
    )
    p.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (2000)")
    p.add_argument("--level", type=float, default=0.95, help="interval level (0.95)")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed (0)")
    p.add_argument("--format", choices=("md", "json"), default="md")


def _setup(args: argparse.Namespace) -> tuple[Tokenizer, Options]:
    tok = Tokenizer(args.tokenizer, lower=not args.keep_case)
    opt = Options(
        span_k=args.span_k, min_span=args.min_span, lead=args.lead, algorithm=args.algorithm
    )
    return tok, opt


def _load(
    args: argparse.Namespace, output_field: str | None = None, path: str | None = None
) -> list[Record]:
    recs = read_records(
        path or args.path, args.source, output_field or args.output, args.id_field, args.group_field
    )
    if args.limit is not None:
        recs = recs[: args.limit]
    return recs


def _score_all(records: Sequence[Record], tok: Tokenizer, opt: Options) -> list[RecordScore]:
    return [score_tokens(tok(r.source), tok(r.output), opt) for r in records]


def _groups(records: Sequence[Record]) -> list[str] | None:
    if records and records[0].group is not None:
        return [r.group for r in records]  # type: ignore[misc]
    return None


def _boot(args: argparse.Namespace) -> BootstrapSettings:
    return BootstrapSettings(n_boot=args.boot, level=args.level, seed=args.seed)


def _fmt(v: object) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _write_json(obj: object) -> None:
    json.dump(obj, sys.stdout, indent=2)
    sys.stdout.write("\n")


def cmd_score(args: argparse.Namespace) -> int:
    tok, opt = _setup(args)
    records = _load(args)
    rows = []
    for rec, r in zip(records, _score_all(records, tok, opt), strict=True):
        d: dict = {"id": rec.id}
        if rec.group is not None:
            d["group"] = rec.group
        d.update(r.to_dict(with_spans=args.spans))
        rows.append(d)
    out = sys.stdout
    if args.format == "jsonl":
        for d in rows:
            out.write(json.dumps(d) + "\n")
    elif args.format == "json":
        _write_json({"tokenizer": tok.describe(), "records": rows})
    else:
        cols = [
            "id",
            "n_source",
            "n_output",
            "coverage",
            "density",
            "compression",
            "novel_2",
            "longest_span",
            "flags",
        ]
        out.write(f"Tokeniser: {tok.describe()}.\n\n")
        out.write("| " + " | ".join(cols) + " |\n")
        out.write("|" + "---|" * len(cols) + "\n")
        for d in rows:
            cells = [", ".join(d[c]) if c == "flags" else _fmt(d[c]) for c in cols]
            out.write("| " + " | ".join(cells) + " |\n")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    tok, opt = _setup(args)
    records = _load(args)
    scores = _score_all(records, tok, opt)
    groups = _groups(records)
    meta = summarize(scores, groups, opt, tuple(args.full or (0.9, 1.0)), _boot(args))
    meta["tokenizer"] = tok.describe()
    meta["group_field"] = args.group_field
    if args.format == "json":
        _write_json(meta)
    else:
        sys.stdout.write(summary_markdown(meta))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    tok, opt = _setup(args)
    if (args.other is None) == (args.other_file is None):
        raise ValueError("give exactly one of --other FIELD or --other-file PATH")
    join = None
    if args.other is not None:
        rec_a = _load(args)
        rec_b = _load(args, output_field=args.other)
        label_a, label_b = args.output, args.other
    else:
        if not args.id_field:
            raise ValueError("--other-file needs --id to pair records")
        field_b = args.other_output or args.output
        all_a = _load(args)
        all_b = read_records(args.other_file, None, field_b, args.id_field, None)
        by_id_b = {r.id: r for r in all_b}
        if len(by_id_b) != len(all_b) or len({r.id for r in all_a}) != len(all_a):
            raise ValueError("ids must be unique in both files")
        rec_a = [r for r in all_a if r.id in by_id_b]
        # The source always comes from the first file.
        rec_b = [Record(r.id, r.source, by_id_b[r.id].output, r.group) for r in rec_a]
        ids_a = {r.id for r in all_a}
        join = {
            "matched": len(rec_a),
            "only_a": len(all_a) - len(rec_a),
            "only_b": sum(1 for r in all_b if r.id not in ids_a),
        }
        label_a, label_b = f"{args.path}:{args.output}", f"{args.other_file}:{field_b}"
    sa, sb = _score_all(rec_a, tok, opt), _score_all(rec_b, tok, opt)
    meta = compare(sa, sb, _groups(rec_a), tuple(args.full or (0.9, 1.0)), _boot(args))
    meta.update(
        {
            "tokenizer": tok.describe(),
            "group_field": args.group_field,
            "groups": len(set(_groups(rec_a) or [])) or None,
            "label_a": label_a,
            "label_b": label_b,
            "join": join,
            "options": {
                "span_k": opt.span_k,
                "min_span": opt.min_span,
                "lead": opt.lead,
                "algorithm": opt.algorithm,
                "n_boot": args.boot,
                "level": args.level,
                "seed": args.seed,
            },
        }
    )
    if args.format == "json":
        _write_json(meta)
    else:
        sys.stdout.write(compare_markdown(meta))
    return 0


def cmd_flag(args: argparse.Namespace) -> int:
    tok, opt = _setup(args)
    records = _load(args)
    wanted = set(args.only.split(",")) if args.only else set(FLAG_NAMES)
    unknown = wanted - set(FLAG_NAMES)
    if unknown:
        raise ValueError(f"unknown flag(s) {sorted(unknown)}; choose from {FLAG_NAMES}")
    rows = []
    for rec, s in zip(records, _score_all(records, tok, opt), strict=True):
        hit = [f for f in s.flags if f in wanted]
        if hit:
            rows.append(
                {
                    "id": rec.id,
                    "flags": hit,
                    "coverage": s.coverage,
                    "longest_span": s.longest_span,
                    "n_output": s.n_output,
                    "n_source": s.n_source,
                }
            )
    if args.format == "json":
        _write_json({"tokenizer": tok.describe(), "records": len(records), "flagged": rows})
    elif args.format == "jsonl":
        for r in rows:
            sys.stdout.write(json.dumps(r) + "\n")
    else:
        sys.stdout.write(flags_markdown(rows, tok.describe()))
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

    p = sub.add_parser("summary", help="dataset distributions, intervals and tails")
    _common(p)
    _boot_args(p)
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("compare", help="paired comparison of two outputs for the same sources")
    _common(p)
    p.add_argument("--other", help="second output field in the same file")
    p.add_argument("--other-file", help="second file, paired by --id")
    p.add_argument("--other-output", help="output field in --other-file (default: --output)")
    _boot_args(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("flag", help="list records by flag for reading")
    _common(p)
    p.add_argument("--only", help=f"comma-separated subset of {','.join(FLAG_NAMES)}")
    p.add_argument("--format", choices=("md", "json", "jsonl"), default="md")
    p.set_defaults(func=cmd_flag)
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
