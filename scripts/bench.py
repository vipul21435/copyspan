"""Time the suffix-automaton fragments against the direct search, and compare the
exact and published (newsroom) procedures, on a JSON lines file.

uv run python scripts/bench.py FILE --source source --output reference [--n 300]
"""

from __future__ import annotations

import argparse
import json
import time

from copyspan import Tokenizer, fragments, fragments_bruteforce, fragments_newsroom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=300, help="records timed (first N)")
    args = ap.parse_args()
    tok = Tokenizer()
    pairs = []
    with open(args.path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            pairs.append((tok(r[args.source]), tok(r[args.output])))

    timed = pairs[: args.n]
    t0 = time.perf_counter()
    fast = [fragments(s, o) for s, o in timed]
    t1 = time.perf_counter()
    slow = [fragments_bruteforce(s, o) for s, o in timed]
    t2 = time.perf_counter()
    same = sum(a == b for a, b in zip(fast, slow, strict=True))
    src_tokens = sum(len(s) for s, _ in timed)
    print(f"records timed: {len(timed)}, mean source length {src_tokens / len(timed):.0f} tokens")
    print(f"suffix automaton: {1000 * (t1 - t0) / len(timed):.2f} ms per record")
    print(f"direct search:    {1000 * (t2 - t1) / len(timed):.2f} ms per record")
    print(f"identical fragments: {same}/{len(timed)}")

    # A long case: the first 30 sources joined, against their 30 outputs joined.
    long_src = [t for s, _ in pairs[:30] for t in s]
    long_out = [t for _, o in pairs[:30] for t in o]
    t0 = time.perf_counter()
    a = fragments(long_src, long_out)
    t1 = time.perf_counter()
    b = fragments_bruteforce(long_src, long_out)
    t2 = time.perf_counter()
    print(
        f"long case: source {len(long_src)} tokens, output {len(long_out)} tokens: "
        f"suffix automaton {1000 * (t1 - t0):.0f} ms, direct search {1000 * (t2 - t1):.0f} ms, "
        f"identical {a == b}"
    )

    differ = 0
    cov_e = cov_n = den_e = den_n = 0.0
    for s, o in pairs:
        if not o:
            continue
        e, n = fragments(s, o), fragments_newsroom(s, o)
        differ += e != n
        cov_e += sum(f.length for f in e) / len(o)
        cov_n += sum(f.length for f in n) / len(o)
        den_e += sum(f.length**2 for f in e) / len(o)
        den_n += sum(f.length**2 for f in n) / len(o)
    k = sum(1 for _, o in pairs if o)
    print(f"all {k} records: exact vs newsroom fragments differ in {differ}")
    print(f"mean coverage exact {cov_e / k:.4f}, newsroom {cov_n / k:.4f}")
    print(f"mean density  exact {den_e / k:.4f}, newsroom {den_n / k:.4f}")


if __name__ == "__main__":
    main()
