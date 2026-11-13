import pytest

from copyspan.tokenize import Tokenizer


def test_word_mode_splits_punctuation_and_folds_case():
    tok = Tokenizer("word", lower=True)
    assert tok("The cat's mat, re-used.") == ["the", "cat's", "mat", ",", "re-used", "."]


def test_whitespace_mode_keeps_punctuation_attached():
    tok = Tokenizer("whitespace", lower=False)
    assert tok("The cat,  sat.") == ["The", "cat,", "sat."]


def test_empty_and_none():
    tok = Tokenizer()
    assert tok("") == []
    assert tok(None) == []
    assert tok("   ") == []


def test_unknown_mode():
    with pytest.raises(ValueError):
        Tokenizer("bpe")


def test_describe_names_mode_and_case():
    assert "case folded" in Tokenizer("word", True).describe()
    assert "whitespace" in Tokenizer("whitespace", False).describe()
    assert "case kept" in Tokenizer("whitespace", False).describe()


def test_public_api_round_trip():
    import copyspan

    tok = copyspan.Tokenizer()
    r = copyspan.score_tokens(tok("The cat sat."), tok("The cat."), copyspan.Options())
    s = copyspan.summarize([r], boot=copyspan.BootstrapSettings(n_boot=10))
    assert s["metrics"]["coverage"]["mean"] == pytest.approx(1.0)
    assert copyspan.__version__ == "0.1.0"
