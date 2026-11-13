"""copyspan: measure how much of an output text is copied from its source."""

from copyspan.fragments import (
    Fragment,
    fragments,
    fragments_bruteforce,
    fragments_newsroom,
    longest_matches,
)
from copyspan.tokenize import Tokenizer

__all__ = [
    "Fragment",
    "Tokenizer",
    "fragments",
    "fragments_bruteforce",
    "fragments_newsroom",
    "longest_matches",
]

__version__ = "0.1.0"
