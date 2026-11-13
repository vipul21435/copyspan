"""Extractive fragments: shared token sequences between a source and an output.

Grusky, Naaman and Artzi (2018) define the extractive fragments of a summary
greedily: walk the summary left to right; at each position take the longest
token sequence that also appears in the article, record it, and continue after
it; a token with no match is skipped.

Three implementations are given:

* ``fragments`` (default): exact longest match at each output position, found in
  linear time with a suffix automaton of the reversed source.
* ``fragments_bruteforce``: the same definition by direct search, quadratic in
  the lengths; used as the reference in tests.
* ``fragments_newsroom``: the pseudocode of the paper as printed. Its inner loop
  jumps past a match in the source (``j <- j'``), so a longer match that starts
  inside an earlier, shorter one can be missed. It exists for comparison.

Ties between equally long matches go to the leftmost source position in all
three.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Fragment:
    """A copied span.

    ``output[out_start : out_start + length] == source[src_start : src_start + length]``
    """

    out_start: int
    src_start: int
    length: int

    def to_dict(self) -> dict[str, int]:
        return {"out_start": self.out_start, "src_start": self.src_start, "length": self.length}


class _SuffixAutomaton:
    """Suffix automaton of a token sequence, with the largest end position per state."""

    __slots__ = ("next", "link", "length", "maxend")

    def __init__(self, seq: Sequence[str]) -> None:
        nxt: list[dict[str, int]] = [{}]
        link = [-1]
        length = [0]
        first = [-1]
        clone_flag = [False]
        last = 0
        for pos, c in enumerate(seq):
            cur = len(length)
            nxt.append({})
            link.append(-1)
            length.append(length[last] + 1)
            first.append(pos)
            clone_flag.append(False)
            p = last
            while p != -1 and c not in nxt[p]:
                nxt[p][c] = cur
                p = link[p]
            if p == -1:
                link[cur] = 0
            else:
                q = nxt[p][c]
                if length[p] + 1 == length[q]:
                    link[cur] = q
                else:
                    clone = len(length)
                    nxt.append(dict(nxt[q]))
                    link.append(link[q])
                    length.append(length[p] + 1)
                    first.append(first[q])
                    clone_flag.append(True)
                    while p != -1 and nxt[p].get(c) == q:
                        nxt[p][c] = clone
                        p = link[p]
                    link[q] = clone
                    link[cur] = clone
            last = cur
        # The end positions of a state are the first-end positions of the
        # non-clone states in its suffix-link subtree; propagate the maximum.
        maxend = [-1 if clone_flag[v] else first[v] for v in range(len(length))]
        buckets: list[list[int]] = [[] for _ in range(length[last] + 1)]
        for v, ln in enumerate(length):
            buckets[ln].append(v)
        for ln in range(len(buckets) - 1, 0, -1):
            for v in buckets[ln]:
                parent = link[v]
                if maxend[v] > maxend[parent]:
                    maxend[parent] = maxend[v]
        self.next = nxt
        self.link = link
        self.length = length
        self.maxend = maxend


def longest_matches(source: Sequence[str], output: Sequence[str]) -> tuple[list[int], list[int]]:
    """For each output position i, the longest prefix of ``output[i:]`` found in ``source``.

    Returns ``(lengths, starts)``: ``lengths[i]`` is that length (0 when
    ``output[i]`` is not in the source) and ``starts[i]`` the leftmost source
    position where it occurs (-1 when the length is 0). Runs in
    O(len(source) + len(output)) dictionary operations.
    """
    m = len(output)
    lengths = [0] * m
    starts = [-1] * m
    if m == 0 or len(source) == 0:
        return lengths, starts
    n = len(source)
    sam = _SuffixAutomaton(list(reversed(source)))
    nxt, link, slen, maxend = sam.next, sam.link, sam.length, sam.maxend
    v, ln = 0, 0
    for i in range(m - 1, -1, -1):
        c = output[i]
        while v != 0 and c not in nxt[v]:
            v = link[v]
            ln = slen[v]
        target = nxt[v].get(c)
        if target is None:
            v, ln = 0, 0
        else:
            v = target
            ln += 1
        lengths[i] = ln
        if ln:
            # A match ending at reversed position e starts at n - 1 - e in the
            # source; the largest e gives the leftmost start.
            starts[i] = n - 1 - maxend[v]
    return lengths, starts


def _greedy(lengths: list[int], starts: list[int]) -> list[Fragment]:
    out: list[Fragment] = []
    i = 0
    m = len(lengths)
    while i < m:
        ln = lengths[i]
        if ln > 0:
            out.append(Fragment(i, starts[i], ln))
            i += ln
        else:
            i += 1
    return out


def fragments(source: Sequence[str], output: Sequence[str]) -> list[Fragment]:
    """Greedy extractive fragments of ``output`` with respect to ``source`` (linear time)."""
    return _greedy(*longest_matches(source, output))


def fragments_bruteforce(source: Sequence[str], output: Sequence[str]) -> list[Fragment]:
    """Same result as :func:`fragments` by direct search over every source position."""
    m, n = len(output), len(source)
    out: list[Fragment] = []
    i = 0
    while i < m:
        best_len, best_j = 0, -1
        for j in range(n):
            ln = 0
            while i + ln < m and j + ln < n and output[i + ln] == source[j + ln]:
                ln += 1
            if ln > best_len:
                best_len, best_j = ln, j
        if best_len:
            out.append(Fragment(i, best_j, best_len))
            i += best_len
        else:
            i += 1
    return out


def fragments_newsroom(source: Sequence[str], output: Sequence[str]) -> list[Fragment]:
    """The greedy procedure exactly as printed in Grusky et al. (2018), Algorithm 1.

    After a match the source index jumps to the end of that match, so the
    result can contain shorter fragments than :func:`fragments`.
    """
    m, n = len(output), len(source)
    out: list[Fragment] = []
    i = 0
    while i < m:
        best_len, best_j = 0, -1
        j = 0
        while j < n:
            if output[i] == source[j]:
                i2, j2 = i, j
                while i2 < m and j2 < n and output[i2] == source[j2]:
                    i2 += 1
                    j2 += 1
                if i2 - i > best_len:
                    best_len, best_j = i2 - i, j
                j = j2
            else:
                j += 1
        if best_len:
            out.append(Fragment(i, best_j, best_len))
            i += best_len
        else:
            i += 1
    return out
