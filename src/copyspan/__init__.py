"""copyspan: measure how much of an output text is copied from its source."""

from copyspan.fragments import (
    Fragment,
    fragments,
    fragments_bruteforce,
    fragments_newsroom,
    longest_matches,
)
from copyspan.metrics import Options, RecordScore, score_tokens
from copyspan.summary import BootstrapSettings, compare, summarize
from copyspan.tokenize import Tokenizer

__all__ = [
    "BootstrapSettings",
    "Fragment",
    "Options",
    "RecordScore",
    "Tokenizer",
    "compare",
    "fragments",
    "fragments_bruteforce",
    "fragments_newsroom",
    "longest_matches",
    "score_tokens",
    "summarize",
]

__version__ = "0.1.0"
