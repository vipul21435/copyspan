# copyspan

Measure how much of an output text is copied from its source, in which spans and from where.

For pairs of source and output text (a document and its summary, an instruction's input and its answer, a context
and a model's response), copyspan finds the extractive fragments of the output, the shared token sequences found
greedily as in Grusky, Naaman and Artzi (2018), and reports coverage, density, compression, novel n-gram shares,
the longest copied span and a few flags per record.

Status: early. Per-record scoring (`copyspan score`) works; dataset summaries, comparisons and the flag listing are
not written yet.

## Install

```
pip install git+https://github.com/vipul21435/copyspan
```

## Quickstart

```
copyspan score examples/pairs.jsonl --source document --output summary --id id --format md
```

```
Tokeniser: word regex (words and single punctuation marks), case folded.

| id | n_source | n_output | coverage | density | compression | novel_2 | longest_span | flags |
|---|---|---|---|---|---|---|---|---|
| r1 | 33 | 15 | 1.000 | 15.000 | 2.200 | 0.000 | 15 | slice |
| r2 | 29 | 14 | 0.429 | 0.429 | 2.071 | 1.000 | 1 |  |
| r3 | 31 | 11 | 0.727 | 2.364 | 2.818 | 0.500 | 4 |  |
| r4 | 23 | 11 | 1.000 | 11.000 | 2.091 | 0.000 | 11 | prefix |
| r5 | 16 | 26 | 0.231 | 0.308 | 0.615 | 0.960 | 2 | no_shared_content, longer_than_source |
| r6 | 12 | 2 | 0.500 | 0.500 | 6.000 | 1.000 | 1 | no_shared_content |
```

The fragments are found with a suffix automaton of the source in linear time and checked in the tests against a
direct quadratic search on random texts.

## Licence

MIT.

Built by Vipul Raj Jha · https://vipul21435.github.io
