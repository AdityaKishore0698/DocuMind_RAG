"""Consolidated human review: build (no overwrite, blinding), validation, summary, per-run export."""

import csv
import json
from types import SimpleNamespace

import pytest

import human_review as HR

QUESTIONS = [
    {"id": "q1", "type": "method", "question": "Q1?", "status": "approved", "gold_answer": "G1", "papers": ["octo"],
     "evidence": [{"paper": "octo", "page": 1, "quote": "diffusion head", "facet": "method"}]},
    {"id": "x1", "type": "cross_paper", "question": "X1?", "status": "approved", "gold_answer": "GX",
     "papers": ["octo", "pi0"], "evidence": [{"paper": "pi0", "page": 2, "quote": "flow matching", "facet": "method"}]},
]


def _trace(qid, answer, graph=False):
    ctx = [{"chunk_id": 1, "paper_id": "octo", "pages": [1], "text": "Octo uses a diffusion head."}]
    t = {"question_id": qid, "outcome": "answered", "answer": answer, "retrieved": ctx}
    if graph:
        t["graph"] = {"citations": {"valid_sources": [1], "invalid_sources": [], "markers": 1, "cited_chunk_ids": [1]}}
        t["retrieved"] = [{**ctx[0], "source_label": "octo.pdf"}]
    return t


@pytest.fixture
def env(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for run_id, graph in (("R_L", False), ("R_CL", True)):
        (runs / run_id).mkdir(parents=True)
        lines = [_trace("q1", f"old answer {run_id}"), {**_trace("q1", f"answer {run_id}", graph)},
                 _trace("x1", f"cross {run_id}", graph)]
        (runs / run_id / "traces.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    monkeypatch.setattr(HR, "RUNS_DIR", runs)
    monkeypatch.setattr(HR, "REVIEW_DIR", tmp_path / "human_review")
    monkeypatch.setattr(HR, "load_questions", lambda: [dict(q) for q in QUESTIONS])
    return tmp_path


def _build(env, **kw):
    args = SimpleNamespace(run=["L=R_L", "CL=R_CL"], blind=False, seed=1, cross_paper_first=False,
                           out=str(env / "review.csv"), **kw)
    HR.cmd_build(args)
    return env / "review.csv"


def _rows(path):
    return list(csv.DictReader(path.open()))


def test_build_uses_final_trace_and_leaves_scores_blank(env):
    rows = _rows(_build(env))
    assert [(r["question_id"], r["system"]) for r in rows] == [("q1", "L"), ("q1", "CL"), ("x1", "L"), ("x1", "CL")]
    assert rows[0]["answer"] == "answer R_L"                       # latest record, not the earlier attempt
    assert all(r[f] == "" for r in rows for f in HR.HUMAN_FIELDS)  # nothing pre-scored
    assert rows[0]["citations"].startswith("n/a") and "cited sources [1]" in rows[1]["citations"]
    assert rows[0]["auto_gold_quote_in_context"] == "1.00" and rows[1]["auto_citation_validity"] == "1.00"
    assert '"diffusion head"' in rows[0]["gold_evidence"] and rows[2]["cross_paper"] == "yes"
    assert "[S1] octo.pdf" in rows[1]["retrieved_context"]


def test_build_never_overwrites(env):
    path = _build(env)
    path.write_text(path.read_text().replace(",,,,,,,", ",2,2,2,NA,1,NA,me,"))  # simulate annotations
    before = path.read_text()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        _build(env)
    assert path.read_text() == before


def test_build_refuses_unfinished_runs(env):
    lines = (HR.RUNS_DIR / "R_L" / "traces.jsonl").read_text().splitlines()
    bad = json.loads(lines[-1]) | {"outcome": "incomplete"}
    (HR.RUNS_DIR / "R_L" / "traces.jsonl").write_text("\n".join(lines[:-1] + [json.dumps(bad)]) + "\n")
    with pytest.raises(SystemExit, match="not reviewable"):
        _build(env)


def test_blind_mode_hides_systems_and_key_restores_them(env):
    args = SimpleNamespace(run=["L=R_L", "CL=R_CL"], blind=True, seed=7, cross_paper_first=True,
                           out=str(env / "blind.csv"))
    HR.cmd_build(args)
    rows = _rows(env / "blind.csv")
    assert {r["system"] for r in rows} <= set("AB") and all(r["run_id"] == "" for r in rows)
    assert rows[0]["question_id"] == "x1"                          # cross-paper first
    restored = HR.read_scored(env / "blind.csv")
    assert {r["system"] for r in restored} == {"L", "CL"} and {r["run_id"] for r in restored} == {"R_L", "R_CL"}


def _score(path, values):
    rows = _rows(path)
    for r in rows:
        r.update(values.get(r["review_id"], {}))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HR.COLUMNS)
        w.writeheader()
        w.writerows(rows)


def test_validation_rejects_out_of_scale_values(env):
    path = _build(env)
    _score(path, {"q1|L": {"correctness": "3"}, "q1|CL": {"refusal_appropriate": "NA"}})
    with pytest.raises(SystemExit, match="(?s)correctness='3'.*refusal_appropriate='NA'"):
        HR.read_scored(path)


def test_summary_normalises_and_pairs(env):
    path = _build(env)
    _score(path, {
        "q1|L": {"correctness": "1", "evidence_support": "2", "refusal_appropriate": "1", "citation_quality": "NA"},
        "q1|CL": {"correctness": "2", "evidence_support": "2", "refusal_appropriate": "1", "citation_quality": "1"},
        "x1|L": {"correctness": "0", "paper_attribution": "0"},
        "x1|CL": {"correctness": "2", "paper_attribution": "1", "completeness": "na"},
    })
    s = HR.summarize(HR.read_scored(path))
    assert s["systems"]["CL"]["correctness"] == {"mean": 1.0, "n": 2}
    assert s["systems"]["L"]["correctness"] == {"mean": 0.25, "n": 2}
    assert s["systems"]["L"]["citation_quality"] == {"mean": None, "n": 0}     # NA is excluded, not zero
    assert s["cross_paper"]["CL"]["paper_attribution"] == {"mean": 1.0, "n": 1}
    assert s["paired_correctness"]["CL vs L"]["better"] == 2


def test_export_run_scores_feeds_report_and_never_overwrites(env):
    path = _build(env)
    _score(path, {"q1|L": {"correctness": "2"}, "x1|L": {"correctness": "1", "paper_attribution": "NA"}})
    HR.cmd_export_run_scores(SimpleNamespace(file=str(path)))
    exported = _rows(HR.RUNS_DIR / "R_L" / "human_scores.csv")
    assert [(r["question_id"], r["correctness"], r["paper_attribution"]) for r in exported] == [
        ("q1", "2", ""), ("x1", "1", "")]
    assert not (HR.RUNS_DIR / "R_CL" / "human_scores.csv").exists()  # nothing scored for CL
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        HR.cmd_export_run_scores(SimpleNamespace(file=str(path)))
