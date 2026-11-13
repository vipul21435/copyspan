"""Tokenisers. Every report states which one was used."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Words (letters, digits, underscore, inner apostrophes and hyphens) or single
# punctuation characters.
_WORD_RE = re.compile(r"\w+(?:['’-]\w+)*|[^\w\s]", re.UNICODE)

MODES = ("whitespace", "word")


@dataclass(frozen=True)
class Tokenizer:
    """Split text into tokens.

    mode: "whitespace" splits on runs of white space; "word" keeps words and
    single punctuation characters as separate tokens.
    lower: fold case (str.casefold) before splitting.
    """

    mode: str = "word"
    lower: bool = True

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown tokeniser mode {self.mode!r}; choose from {MODES}")

    def __call__(self, text: str | None) -> list[str]:
        if not text:
            return []
        if self.lower:
            text = text.casefold()
        if self.mode == "whitespace":
            return text.split()
        return _WORD_RE.findall(text)

    def describe(self) -> str:
        case = "case folded" if self.lower else "case kept"
        if self.mode == "whitespace":
            return f"whitespace split, {case}"
        return f"word regex (words and single punctuation marks), {case}"
