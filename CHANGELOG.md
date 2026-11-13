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
- Input checks: a named `--id` field must be present in every row; files with a byte-order mark are read
  correctly; float, boolean and object field values raise an error instead of being turned into text; a negative
  `--limit` and `--full` thresholds outside (0, 1] are rejected.
- Content words are compared after stripping surrounding punctuation, and contractions of function words count as
  function words.
- Unclustered tail shares use the Wilson score interval; `summary` and `compare` warn when clustering on fewer
  than 20 groups.
- README corrections: the published unigram shares lie above both tokenisers, the published compression ratio is
  for the training set, the licence sentence is quoted in full, and the record counts behind the Dolly n-gram
  means are given.
