"""Read records from JSON lines or CSV."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Record:
    id: str
    source: str
    output: str
    group: str | None = None


def _json_lines(lines: Iterable[str], name: str) -> Iterator[dict]:
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name}:{lineno}: not valid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{name}:{lineno}: expected a JSON object")
        yield row


def _rows(path: str | Path) -> Iterator[dict]:
    p = str(path)
    if p == "-":
        yield from _json_lines(sys.stdin, "<stdin>")
        return
    if p.endswith(".csv") or p.endswith(".tsv"):
        delim = "\t" if p.endswith(".tsv") else ","
        csv.field_size_limit(sys.maxsize)
        with open(p, newline="", encoding="utf-8") as fh:
            yield from csv.DictReader(fh, delimiter=delim)
        return
    with open(p, encoding="utf-8") as fh:
        yield from _json_lines(fh, p)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_text(v) for v in value)
    return str(value)


def read_records(
    path: str | Path,
    source_field: str | None,
    output_field: str,
    id_field: str | None = None,
    group_field: str | None = None,
) -> list[Record]:
    """Load records. Rows missing the source or output field raise ``ValueError``.

    A list value (for example a list of sentences) is joined with newlines.
    Without ``id_field`` the id is the 0-based row number. With ``source_field``
    set to ``None`` only the output is read (the source is left empty).
    """
    out: list[Record] = []
    for k, row in enumerate(_rows(path)):
        for f in (source_field, output_field):
            if f is not None and f not in row:
                raise ValueError(f"row {k}: field {f!r} not found (fields: {sorted(row)})")
        rid = str(row[id_field]) if id_field and row.get(id_field) is not None else str(k)
        grp = None
        if group_field:
            if row.get(group_field) is None:
                raise ValueError(f"row {k}: group field {group_field!r} missing")
            grp = str(row[group_field])
        src = _text(row[source_field]) if source_field is not None else ""
        out.append(Record(rid, src, _text(row[output_field]), grp))
    return out
