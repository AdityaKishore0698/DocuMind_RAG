"""LLM-judge pipeline: blinding, validation/NA rules, retries, second pass, finalisation, preservation."""

import csv
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

import human_review as HR
import llm_judge as J

ROWS = [
    # review_id, type, cross, system, citations
    ("q1|A", "method", "no", "A", "n/a (system is not prompted to cite)"),
    ("q1|B", "method", "no", "B", "cited sources [1]; invalid markers []; markers 1; cited chunk ids [5]"),
    ("x1|A", "cross_paper", "yes", "A", "n/a (system is not prompted to cite)"),
    ("neg1|A", "negative", "no", "A", "n/a (system is not prompted to cite)"),
]


def _review_row(rid, qtype, cross, system, citations):
    base = {c: "" for c in HR.COLUMNS}
    base.update({"review_id": rid, "question_id": rid.split("|")[0], "question_type": qtype, "cross_paper": cross,
                 "question": "Q?", "gold_answer": "G", "gold_evidence": "- p p.1: \"g\"", "system": system,
                 "answer": "ans [S1]" if "markers 1" in citations else "ans", "retrieved_context": "[S1] ctx",
                 "citations": citations, "auto_gold_quote_in_context": "1.00", "auto_citation_validity": "1.00"})
    return base


@pytest.fixture
def sheet(tmp_path, monkeypatch):
    review = tmp_path / "review_X_blind.csv"
    with review.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HR.COLUMNS)
        w.writeheader()
        w.writerows(_review_row(*r) for r in ROWS)
    key = tmp_path / "review_X_blind_KEY.json"
    key.write_text(json.dumps({"key": {"q1|A": {"system": "L", "run_id": "R_L"}, "q1|B": {"system": "CL", "run_id": "R_CL"},
                                       "x1|A": {"system": "L", "run_id": "R_L"}, "neg1|A": {"system": "L", "run_id": "R_L"}}}))
    rubric = tmp_path / "HUMAN_REVIEW.md"
    rubric.write_text("intro mentions _KEY.json\n\n| Field | Scale | Definition |\n|---|---|---|\n| x | y | z |\n\n"
                      "## Workflow\nkey commands\n")
    monkeypatch.setattr(J, "RUBRIC_SOURCE", rubric)
    return review, key


def _prepare(review, batch_size=2):
    J.cmd_prepare(SimpleNamespace(review=str(review), batch_size=batch_size))
    return next(review.parent.glob("judge_*"))


def _field(score, uncertain=False, evidence="quote"):
    return {"score": score, "rationale": "because", "evidence": evidence if score != "NA" else "", "uncertain": uncertain}


def good(rid, **override):
    qtype = {r[0]: r[1] for r in ROWS}[rid]
    cross = {r[0]: r[2] for r in ROWS}[rid] == "yes"
    graph = "markers 1" in {r[0]: r[4] for r in ROWS}[rid]
    fields = {
        "correctness": _field("NA" if qtype == "negative" else "2"),
        "evidence_support": _field("2"),
        "completeness": _field("NA" if qtype == "negative" else "1"),
        "citation_quality": _field("1" if graph else "NA"),
        "refusal_appropriate": _field("1"),
        "paper_attribution": _field("1" if cross else "NA"),
    }
    fields.update(override)
    return {"review_id": rid, "fields": fields}


def _write_out(judge_dir, name, judgments):
    path = judge_dir / "outputs" / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join((j if isinstance(j, str) else json.dumps(j)) + "\n" for j in judgments))


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --- blinding & preservation ----------------------------------------------------------------------
def test_prepare_inputs_are_blinded_and_key_is_never_read(sheet):
    review, key = sheet
    before = (_sha(review), key.stat().st_size, key.stat().st_mtime)
    os.chmod(key, 0)                                   # unreadable: prepare must only stat() it
    try:
        jd = _prepare(review)
    finally:
        os.chmod(key, 0o644)
    assert (_sha(review), key.stat().st_size, key.stat().st_mtime) == before
    lines = [json.loads(line) for p in sorted((jd / "inputs").glob("*.jsonl")) for line in p.read_text().splitlines()]
    assert len(lines) == 4 and all(set(r) == set(J.JUDGE_FIELDS) for r in lines)
    blob = "".join(p.read_text() for p in (jd / "inputs").glob("*"))
    assert "R_L" not in blob and "auto_" not in blob and '"CL"' not in blob
    rubric = (jd / "rubric.md").read_text()
    assert rubric.startswith("| Field |") and "_KEY" not in rubric and "Workflow" not in rubric
    manifest = json.loads((jd / "manifest.json").read_text())
    assert manifest["prompt_version"] == "v1" and manifest["key_stat"]["size"] == before[1]


def test_prepare_refuses_unblinded_or_prescored_sheets(sheet, tmp_path):
    review, _ = sheet
    rows = list(csv.DictReader(review.open()))
    for mutate, msg in ((lambda r: r.update(run_id="R_L"), "not blinded"), (lambda r: r.update(correctness="2"), "already contains")):
        bad = tmp_path / f"bad_{msg[:3]}.csv"
        changed = [dict(r) for r in rows]
        mutate(changed[0])
        with bad.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=HR.COLUMNS)
            w.writeheader()
            w.writerows(changed)
        with pytest.raises(SystemExit, match=msg):
            J.cmd_prepare(SimpleNamespace(review=str(bad), batch_size=2))


# --- validation / NA rules -----------------------------------------------------------------------------
@pytest.mark.parametrize("rid,override,needle", [
    ("q1|A", {"correctness": _field("3")}, "not in"),
    ("q1|A", {"correctness": _field("NA")}, "NA only allowed for negative"),
    ("neg1|A", {"correctness": _field("1")}, "must be NA for negative"),
    ("q1|A", {"citation_quality": _field("2")}, "must be NA for rows without citations"),
    ("q1|B", {"citation_quality": _field("NA")}, "must be scored"),
    ("q1|A", {"paper_attribution": _field("1")}, "must be NA for non-cross"),
    ("q1|A", {"refusal_appropriate": _field("NA")}, "NA not allowed"),
    ("q1|A", {"evidence_support": {"score": "2", "rationale": "", "evidence": "q", "uncertain": False}}, "empty rationale"),
    ("q1|A", {"correctness": _field("2", evidence="")}, "scored without evidence"),
    ("q1|A", {"correctness": {"score": "2", "rationale": "r", "evidence": "q", "uncertain": "maybe"}}, "true/false"),
])
def test_validation_rules(sheet, rid, override, needle):
    review, _ = sheet
    rows = {r["review_id"]: r for r in csv.DictReader(review.open())}
    assert J.validate_judgment(good(rid), rows[rid]) == []
    problems = J.validate_judgment(good(rid, **override), rows[rid])
    assert any(needle in p for p in problems), problems


def test_missing_fields_object_and_missing_field(sheet):
    review, _ = sheet
    row = next(csv.DictReader(review.open()))
    assert J.validate_judgment({"review_id": "q1|A"}, row) == ["missing 'fields' object"]
    j = good("q1|A")
    del j["fields"]["completeness"]
    assert "completeness: missing" in J.validate_judgment(j, row)


# --- incomplete outputs and retries ----------------------------------------------------------------------
def test_retries_then_unresolved_and_unexpected_lines(sheet):
    review, _ = sheet
    jd = _prepare(review)
    rows = {r["review_id"]: r for r in csv.DictReader(review.open())}
    bad = good("q1|A", correctness=_field("5"))
    _write_out(jd, "pass1_batch_01.jsonl", [bad, good("q1|B"), "not json", {"review_id": "zzz|A", "fields": {}}])
    res = J.collect(jd, "pass1", list(rows), rows)
    assert set(res["valid"]) == {"q1|B"} and set(res["retryable"]) == {"q1|A", "x1|A", "neg1|A"}
    assert len(res["unknown"]) == 2
    J.cmd_retry_batch(SimpleNamespace(judge_dir=str(jd), pass_name="pass1"))
    retry_rows = [json.loads(x)["review_id"] for x in (jd / "inputs" / "pass1_retry1.jsonl").read_text().splitlines()]
    assert retry_rows == ["q1|A", "x1|A", "neg1|A"]
    _write_out(jd, "pass1_retry1.jsonl", [bad, good("x1|A"), good("neg1|A")])
    _write_out(jd, "pass1_retry2.jsonl", [bad])
    res = J.collect(jd, "pass1", list(rows), rows)
    assert res["unresolved"] == ["q1|A"] and res["attempts"]["q1|A"] == 3     # gave up after 3 attempts
    assert set(res["valid"]) == {"q1|B", "x1|A", "neg1|A"}


# --- second pass & finalisation -------------------------------------------------------------------------
def test_second_pass_selection_disagreements_and_unblinding(sheet):
    review, key = sheet
    jd = _prepare(review)
    _write_out(jd, "pass1_batch_01.jsonl", [good("q1|A", correctness=_field("1", uncertain=True)), good("q1|B")])
    _write_out(jd, "pass1_batch_02.jsonl", [good("x1|A"), good("neg1|A")])
    J.cmd_select_second(SimpleNamespace(judge_dir=str(jd)))
    sel = json.loads((jd / "pass2_selection.json").read_text())
    assert sel["rows"] == ["q1|A", "x1|A"]
    assert sel["reasons"] == {"q1|A": ["uncertain"], "x1|A": ["cross_paper"]}
    _write_out(jd, "pass2_batch_01.jsonl", [good("q1|A", correctness=_field("2")),
                                             good("x1|A", paper_attribution=_field("0"))])
    review_hash, key_hash = _sha(review), _sha(key)
    J.cmd_finalize(SimpleNamespace(judge_dir=str(jd)))
    assert (_sha(review), _sha(key)) == (review_hash, key_hash)              # originals untouched
    judged = next(jd.glob("judged_v1_*.csv"))
    rows = {r["review_id"]: r for r in csv.DictReader(judged.open())}
    assert rows["q1|A"]["correctness"] == "1"                                # pass 1 is primary — not averaged
    assert "DISAGREE correctness 1/2" in rows["q1|A"]["notes"] and "uncertain:correctness" in rows["q1|A"]["notes"]
    assert "DISAGREE paper_attribution 1/0" in rows["x1|A"]["notes"]
    assert rows["neg1|A"]["correctness"] == "NA" and rows["q1|A"]["reviewer"] == "AI judge (v1)"
    dis = json.loads(next(jd.glob("disagreements_v1_*.json")).read_text())
    assert {(d["review_id"], d["field"]) for d in dis} == {("q1|A", "correctness"), ("x1|A", "paper_attribution")}
    meta = json.loads(next(jd.glob("judge_run_v1_*.json")).read_text())
    assert "NOT independent models" in meta["pass_independence"] and "not human-verified" in meta["label"]
    audit = [json.loads(x) for x in next(jd.glob("audit_v1_*.jsonl")).read_text().splitlines()]
    assert len(audit) == 6 and all(a["fields"]["correctness"]["rationale"] for a in audit)
    # un-blinding only now, via the existing summary path
    restored = HR.read_scored(judged, key)
    assert {r["system"] for r in restored} == {"L", "CL"}
    summary = HR.summarize(restored)
    assert summary["systems"]["CL"]["correctness"] == {"mean": 1.0, "n": 1}


def test_finalize_refuses_if_original_changed(sheet):
    review, _ = sheet
    jd = _prepare(review)
    review.write_text(review.read_text() + "\n")
    with pytest.raises(SystemExit, match="changed since prepare"):
        J.cmd_finalize(SimpleNamespace(judge_dir=str(jd)))


# --- scoped phase --------------------------------------------------------------------------------------
def test_scope_limits_judging_reuses_pass1_and_marks_the_rest_not_judged(sheet):
    review, key = sheet
    jd = _prepare(review)
    _write_out(jd, "pass1_batch_02.jsonl", [good("x1|A")])               # already judged in an earlier batch
    J.cmd_scope(SimpleNamespace(judge_dir=str(jd), types="cross_paper,negative", batch_size=16, seed=3))
    scope = json.loads((jd / "scope.json").read_text())
    assert scope["rows"] == ["x1|A", "neg1|A"] and scope["excluded"] == 2
    todo = [json.loads(x)["review_id"] for x in (jd / "inputs" / "pass1_scope_batch_01.jsonl").read_text().splitlines()]
    assert todo == ["neg1|A"]                                             # no re-judging of valid pass-1 rows
    p2 = sorted(json.loads(x)["review_id"] for x in (jd / "inputs" / "pass2_batch_01.jsonl").read_text().splitlines())
    assert p2 == ["neg1|A", "x1|A"]                                       # pass 2 covers the whole scope
    _write_out(jd, "pass1_scope_batch_01.jsonl", [good("neg1|A")])
    _write_out(jd, "pass2_batch_01.jsonl", [good("neg1|A"), good("x1|A")])
    J.cmd_finalize(SimpleNamespace(judge_dir=str(jd)))
    rows = {r["review_id"]: r for r in csv.DictReader(next(jd.glob("judged_v1_*.csv")).open())}
    assert rows["q1|A"]["correctness"] == "" and rows["q1|A"]["notes"].startswith("NOT_JUDGED")
    assert rows["x1|A"]["paper_attribution"] == "1" and rows["neg1|A"]["refusal_appropriate"] == "1"
    meta = json.loads(next(jd.glob("judge_run_v1_*.json")).read_text())
    assert meta["scope"]["rows"] == 2 and meta["pass1"]["missing"] == [] and meta["material_disagreements"] == 0
