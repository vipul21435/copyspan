import pytest

from copyspan.metrics import Options, novel_ngram_share, score_tokens


def toks(s):
    return s.split()


def test_hand_worked_record():
    source = toks("the cat sat on the mat and the dog sat on the rug")
    output = toks("the dog sat on the mat today")
    r = score_tokens(source, output)
    # fragments: 5 tokens + 1 token, 7 output tokens
    assert r.coverage == pytest.approx(6 / 7)
    assert r.density == pytest.approx((25 + 1) / 7)
    assert r.compression == pytest.approx(13 / 7)
    assert r.longest_span == 5
    assert r.long_span_share == pytest.approx(5 / 7)
    assert r.n_fragments == 2
    # novel unigrams: "today" only
    assert r.novel[1] == pytest.approx(1 / 7)
    # bigrams of output: the-dog, dog-sat, sat-on, on-the, the-mat, mat-today -> "mat today" novel
    assert r.novel[2] == pytest.approx(1 / 6)
    # one long fragment from source[7:12]: mean position (7 + 2.5) / 13
    assert r.position_mean == pytest.approx(9.5 / 13)
    assert r.lead_share == pytest.approx(0.0)
    assert r.order_kept is None
    assert r.flags == []


def test_identities_and_bounds():
    source = toks("a b c d e f g a b c")
    for output in (toks("a b c"), toks("x y"), toks("a x b y c d e"), toks("g a b c d")):
        r = score_tokens(source, output)
        assert 0.0 <= r.coverage <= 1.0
        assert r.density <= r.coverage * r.longest_span + 1e-12
        assert r.density >= r.coverage or r.coverage == 0
        assert r.long_span_share <= r.coverage + 1e-12
        # every copied token is in the source, so novel unigrams = 1 - coverage
        assert r.novel[1] == pytest.approx(1 - r.coverage)


def test_prefix_slice_flags():
    source = toks("one two three four five")
    assert "prefix" in score_tokens(source, toks("one two three")).flags
    assert "slice" in score_tokens(source, toks("two three four")).flags
    assert score_tokens(source, toks("two four")).flags == []


def test_no_shared_content_and_longer():
    source = toks("the river is wide")
    r = score_tokens(source, toks("the mountain is tall and it is very cold"))
    assert "no_shared_content" in r.flags
    assert "longer_than_source" in r.flags


def test_empty_and_one_token():
    r = score_tokens([], [])
    assert r.coverage is None and r.density is None and r.compression is None
    assert r.flags == ["empty_output"]
    r = score_tokens(toks("a"), toks("a"))
    assert (r.coverage, r.density, r.compression) == pytest.approx((1.0, 1.0, 1.0))
    assert r.novel[2] is None
    r = score_tokens([], toks("word"))
    assert r.coverage == pytest.approx(0.0)
    assert r.compression == pytest.approx(0.0)
    assert "longer_than_source" in r.flags


def test_order_measures():
    source = toks("a b c d e f g h i j k l")
    kept = score_tokens(source, toks("a b c x g h i"))
    assert kept.order_kept is True and kept.order_concordance == pytest.approx(1.0)
    swapped = score_tokens(source, toks("g h i x a b c"))
    assert swapped.order_kept is False and swapped.order_concordance == pytest.approx(0.0)


def test_lead_share_full_and_partial():
    source = toks("a b c d e f g h")  # lead 0.25 -> first 2 tokens
    r = score_tokens(source, toks("a b c"))
    assert r.lead_share == pytest.approx(2 / 3)
    r = score_tokens(source, toks("f g h"), Options(lead=0.5))
    assert r.lead_share == pytest.approx(0.0)


def test_novel_share_short_output():
    assert novel_ngram_share(["a"], ["a"], 2) is None


def test_newsroom_algorithm_option():
    source = toks("a a a b")
    output = toks("a a b")
    assert score_tokens(source, output).density == pytest.approx(3.0)
    r = score_tokens(source, output, Options(algorithm="newsroom"))
    assert r.density == pytest.approx((4 + 1) / 3)
    assert r.coverage == pytest.approx(1.0)


def test_options_validation():
    with pytest.raises(ValueError):
        Options(algorithm="fast")
    with pytest.raises(ValueError):
        Options(span_k=0)
    with pytest.raises(ValueError):
        Options(lead=0)


def test_to_dict_keys():
    d = score_tokens(toks("a b c"), toks("a b")).to_dict(with_spans=True)
    assert d["novel_1"] == pytest.approx(0.0)
    assert d["spans"] == [{"out_start": 0, "src_start": 0, "length": 2}]


def test_bounds_on_random_pairs():
    import random

    rng = random.Random(21)
    for _ in range(1000):
        vocab = [str(k) for k in range(rng.randint(1, 5))]
        source = [rng.choice(vocab) for _ in range(rng.randint(0, 30))]
        output = [rng.choice(vocab + ["new"]) for _ in range(rng.randint(1, 20))]
        r = score_tokens(source, output)
        longest_fragment = max((f.length for f in r.spans), default=0)
        assert 0.0 <= r.coverage <= 1.0
        assert r.coverage <= r.density + 1e-12
        assert r.density <= r.coverage * longest_fragment + 1e-12
        assert longest_fragment <= r.longest_span
        assert r.novel[1] == pytest.approx(1 - r.coverage)
        for n in (2, 3, 4):
            if r.novel[n] is not None:
                assert 0.0 <= r.novel[n] <= 1.0
        if r.lead_share is not None:
            assert 0.0 <= r.lead_share <= 1.0 and 0.0 <= r.position_mean <= 1.0


def test_content_words_handle_contractions_and_punctuation():
    from copyspan.metrics import content_key
    from copyspan.tokenize import Tokenizer

    assert content_key("it's") is None
    assert content_key("they\u2019re") is None
    assert content_key("don't") is None
    assert content_key("River,") == "river"
    assert content_key("...") is None
    ws = Tokenizer("whitespace", lower=False)
    # only function words in the output: flagged
    r = score_tokens(ws("The river is wide."), ws("It's what they're for."))
    assert "no_shared_content" in r.flags
    # "River," and "river" share a content word under the whitespace tokeniser
    r = score_tokens(ws("the river is wide"), ws("A River, then."))
    assert "no_shared_content" not in r.flags
