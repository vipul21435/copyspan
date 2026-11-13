"""Download the public datasets used in the README and write seeded samples as JSON lines.

Run with:  uv run --with pyarrow python scripts/prepare_data.py --out DIR

Nothing from the datasets is stored in this repository; the samples are written
to DIR (outside the repository by choice).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

SOURCES = {
    "cnndm-test.parquet": "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/"
    "3.0.0/test-00000-of-00001.parquet",
    "xsum-test.parquet": "https://huggingface.co/api/datasets/EdinburghNLP/xsum/parquet/"
    "default/test/0.parquet",
    "dolly.jsonl": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/"
    "databricks-dolly-15k.jsonl",
}
CONTEXT_CATEGORIES = {"closed_qa", "information_extraction", "summarization"}
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def fetch(out: Path) -> None:
    for name, url in SOURCES.items():
        dest = out / name
        if not dest.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)


def lead3(article: str) -> str:
    return " ".join(_SENT.split(article.strip())[:3])


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000, help="records sampled per summarisation set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fetch(args.out)

    import pyarrow.parquet as pq

    cnn = pq.read_table(args.out / "cnndm-test.parquet").to_pylist()
    idx = sorted(random.Random(args.seed).sample(range(len(cnn)), args.n))
    write(
        args.out / "cnndm-sample.jsonl",
        [
            {
                "id": cnn[k]["id"],
                "source": cnn[k]["article"],
                "reference": cnn[k]["highlights"],
                "lead3": lead3(cnn[k]["article"]),
            }
            for k in idx
        ],
    )
    xsum = pq.read_table(args.out / "xsum-test.parquet").to_pylist()
    idx = sorted(random.Random(args.seed).sample(range(len(xsum)), args.n))
    write(
        args.out / "xsum-sample.jsonl",
        [
            {"id": xsum[k]["id"], "source": xsum[k]["document"], "reference": xsum[k]["summary"]}
            for k in idx
        ],
    )
    rows = []
    with (args.out / "dolly.jsonl").open(encoding="utf-8") as fh:
        for k, line in enumerate(fh):
            r = json.loads(line)
            if r["category"] in CONTEXT_CATEGORIES and r["context"].strip():
                rows.append(
                    {
                        "id": f"row-{k}",
                        "category": r["category"],
                        "source": r["context"],
                        "response": r["response"],
                    }
                )
    write(args.out / "dolly-context.jsonl", rows)


if __name__ == "__main__":
    main()
