import json

import pytest

from copyspan.cli import main
from copyspan.io import read_records


@pytest.fixture
def jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = [
        {
            "id": "a",
            "g": "x",
            "src": "one two three four five",
            "out": "one two three",
            "alt": "six",
        },
        {"id": "b", "g": "x", "src": "red green blue", "out": "green blue red", "alt": "red"},
        {"id": "c", "g": "y", "src": "alpha beta", "out": "", "alt": "alpha beta"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    return p


def test_score_jsonl(jsonl, capsys):
    assert main(["score", str(jsonl), "--source", "src", "--output", "out", "--id", "id"]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [r["id"] for r in lines] == ["a", "b", "c"]
    assert lines[0]["coverage"] == 1.0
    assert "prefix" in lines[0]["flags"]
    assert lines[2]["flags"] == ["empty_output"]


def test_score_md_and_json(jsonl, capsys):
    main(["score", str(jsonl), "--source", "src", "--output", "out", "--format", "md"])
    out = capsys.readouterr().out
    assert out.startswith("Tokeniser: word regex")
    assert "| 0 | 5 | 3 | 1.000 |" in out
    main(
        [
            "score",
            str(jsonl),
            "--source",
            "src",
            "--output",
            "out",
            "--format",
            "json",
            "--spans",
            "--group",
            "g",
            "--tokenizer",
            "whitespace",
            "--keep-case",
        ]
    )
    doc = json.loads(capsys.readouterr().out)
    assert doc["tokenizer"].startswith("whitespace")
    assert doc["records"][0]["group"] == "x"
    assert doc["records"][0]["spans"] == [{"out_start": 0, "src_start": 0, "length": 3}]


def test_missing_field_is_an_error(jsonl, capsys):
    assert main(["score", str(jsonl), "--source", "nope", "--output", "out"]) == 2
    assert "field 'nope' not found" in capsys.readouterr().err


def test_csv_input(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text('id,src,out\n1,"a b c","a b"\n2,"x y",z\n')
    recs = read_records(p, "src", "out", "id")
    assert [r.id for r in recs] == ["1", "2"]
    assert recs[0].source == "a b c"


def test_bad_json_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"a": 1}\nnot json\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        read_records(p, "a", "a")


def test_list_values_are_joined(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps({"s": ["a b", "c"], "o": None}) + "\n")
    rec = read_records(p, "s", "o")[0]
    assert rec.source == "a b\nc" and rec.output == ""


def test_stdin_input(monkeypatch, capsys):
    import io as _io

    monkeypatch.setattr("sys.stdin", _io.StringIO('{"s": "a b c", "o": "a b"}\n\n[1, 2]\n'))
    assert main(["score", "-", "--source", "s", "--output", "o"]) == 2
    assert "<stdin>:3: expected a JSON object" in capsys.readouterr().err
    monkeypatch.setattr("sys.stdin", _io.StringIO('{"s": "a b c", "o": "a b"}\n'))
    assert main(["score", "-", "--source", "s", "--output", "o"]) == 0
    assert json.loads(capsys.readouterr().out)["coverage"] == 1.0


def test_tsv_and_group_missing(tmp_path):
    p = tmp_path / "d.tsv"
    p.write_text("src\tout\tg\na b\ta\tx\nc d\tc\t\n")
    recs = read_records(p, "src", "out")
    assert [r.output for r in recs] == ["a", "c"]
    q = tmp_path / "g.jsonl"
    q.write_text(json.dumps({"src": "a", "out": "a"}) + "\n")
    with pytest.raises(ValueError, match="group field"):
        read_records(q, "src", "out", group_field="g")
