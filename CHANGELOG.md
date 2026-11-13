# Changelog

## 0.1.0 (2026-11-13)

- Greedy extractive fragments (Grusky, Naaman and Artzi 2018) by a suffix automaton of the reversed source, with a
  direct-search reference and the paper's published procedure (`--algorithm newsroom`).
- Per-record coverage, density, compression, novel n-gram shares for n = 1 to 4, longest copied span, share of
  tokens in spans of at least k tokens, source position of copies, lead share and order measures.
- Flags: prefix, slice, no shared content word, longer than source, empty output.
- Dataset summary with quantiles, percentile bootstrap intervals (clustered by a group field when given) and tail
  counts; paired comparison of two outputs for the same sources.
- Commands `copyspan score`, `summary`, `compare` and `flag`; JSON lines or CSV/TSV input; Markdown and JSON output;
  word-regex or whitespace tokenisation with optional case folding, named in every report.
- `scripts/prepare_data.py` and `scripts/bench.py` for the real-data numbers and timings in the README.
