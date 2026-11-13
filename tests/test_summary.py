import json

import pytest

from copyspan.cli import main
from copyspan.metrics import Options, score_tokens
from copyspan.summary import BootstrapSettings, compare, position_histogram, summarize


def toks(s):
    return s.split()


SRC = toks("a b c d e f g h i j")
OUTS = ["a b c d", "x y z", "e f g h i j", "b d f", "q"]


def _scores(outs=OUTS):
    return [score_tokens(SRC, toks(o)) for o in outs]


def test_summary_means_and_tails():
    s = summarize(_scores(), boot=BootstrapSettings(n_boot=200))
    cov = s["metrics"]["coverage"]
    assert cov["n"] == 5
    assert cov["mean"] == pytest.approx((1 + 0 + 1 + 1 + 0) / 5)
    assert cov["ci"][0] <= cov["mean"] <= cov["ci"][1]
    t = s["tails"]
    assert t["records_with_output"] == 5
    assert t["no_token_copied"]["count"] == 2
    assert t["whole_output_one_span"]["count"] == 2
    full = {row["threshold"]: row for row in t["fully_copied"]}
    assert full[1.0]["coverage"]["count"] == 3
    # "b d f" is fully covered by one-token fragments, not by long spans
    assert full[1.0]["long_span_share"]["count"] == 2
    assert s["flags"]["prefix"] == 1


def test_summary_with_groups_and_empty_output():
    scores = _scores(OUTS + [""])
    groups = ["g1", "g1", "g2", "g2", "g3", "g3"]
    s = summarize(scores, groups, boot=BootstrapSettings(n_boot=100))
    assert s["groups"] == 3
    assert s["metrics"]["coverage"]["n"] == 5
    assert s["tails"]["records_with_output"] == 5
    with pytest.raises(ValueError):
        summarize(scores, groups[:2])


def test_position_histogram_sums_to_one():
    hist = position_histogram(_scores(), Options())
    assert sum(hist) == pytest.approx(1.0)
    # "a b c d" covers the first four tenths, "e f g h i j" the last six
    assert hist == pytest.approx([0.1] * 10)
    assert position_histogram([score_tokens([], [])], Options()) == [0.0] * 10


def test_compare_paired_differences():
    a = _scores(["a b c d", "e f g", "x"])
    b = _scores(["a x", "e q r", "x"])
    c = compare(a, b, boot=BootstrapSettings(n_boot=100))
    m = c["metrics"]["coverage"]
    assert m["pairs"] == 3
    assert m["diff"] == pytest.approx(m["mean_a"] - m["mean_b"])
    assert c["tails"]["coverage>=1"]["count_a"] == 2
    assert c["tails"]["coverage>=1"]["count_b"] == 0
    with pytest.raises(ValueError):
        compare(a, b[:2])


def test_compare_identical_outputs_zero_difference():
    a = _scores()
    c = compare(a, a, groups=["x", "x", "y", "y", "z"], boot=BootstrapSettings(n_boot=50))
    for m in c["metrics"].values():
        if m["diff"] is not None:
            assert m["diff"] == 0.0 and m["ci"] == [0.0, 0.0]


@pytest.fixture
def data(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = [
        {"id": f"r{k}", "g": f"g{k % 2}", "src": " ".join(SRC), "out": o, "alt": "a b"}
        for k, o in enumerate(OUTS)
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    q = tmp_path / "other.jsonl"
    q.write_text(
        "\n".join(json.dumps({"id": f"r{k}", "pred": "c d e"}) for k in range(1, 7)) + "\n"
    )
    return p, q


def test_cli_summary_md_and_json(data, capsys):
    p, _ = data
    args = ["summary", str(p), "--source", "src", "--output", "out", "--boot", "100"]
    assert main(args + ["--group", "g", "--full", "0.5"]) == 0
    md = capsys.readouterr().out
    assert "clustered by `g` (2 groups)" in md
    assert "coverage >= 0.5" in md
    assert "Where copies come from" in md
    assert main(args + ["--format", "json", "--limit", "2"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["records"] == 2 and doc["tokenizer"].startswith("word")


def test_cli_compare_same_file_and_other_file(data, capsys):
    p, q = data
    base = ["compare", str(p), "--source", "src", "--output", "out", "--boot", "50"]
    assert main(base + ["--other", "alt"]) == 0
    assert "Output a: `out`; output b: `alt`" in capsys.readouterr().out
    assert (
        main(
            base
            + ["--id", "id", "--other-file", str(q), "--other-output", "pred", "--format", "json"]
        )
        == 0
    )
    doc = json.loads(capsys.readouterr().out)
    assert doc["join"] == {"matched": 4, "only_a": 1, "only_b": 2}
    assert main(base + ["--id", "id", "--other-file", str(q), "--other-output", "pred"]) == 0
    assert "4 matched" in capsys.readouterr().out
    assert main(base) == 2
    assert main(base + ["--other-file", str(q)]) == 2


def test_cli_flag(data, capsys):
    p, _ = data
    base = ["flag", str(p), "--source", "src", "--output", "out", "--id", "id"]
    assert main(base) == 0
    md = capsys.readouterr().out
    assert "| r0 | prefix |" in md and "| r2 | slice |" in md
    assert main(base + ["--only", "no_shared_content", "--format", "json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert {r["id"] for r in doc["flagged"]} == {"r1", "r4"}
    assert main(base + ["--only", "prefix", "--format", "jsonl"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "r0"
    assert main(base + ["--only", "slice", "--limit", "1"]) == 0
    assert "(none)" in capsys.readouterr().out
    assert main(base + ["--only", "bogus"]) == 2
