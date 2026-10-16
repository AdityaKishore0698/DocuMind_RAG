"""AI judging of the blinded human-review sheet (Claude Code subagents as judges).

The judge itself runs outside this script (fresh Claude Code subagents, one per batch). This module does
everything around it, deterministically:

    PY=evaluation/.venv/bin/python
    $PY evaluation/llm_judge.py prepare  evaluation/results/human_review/<review>_blind.csv
    #   → judge_<stamp>/ : rubric.md (verbatim from HUMAN_REVIEW.md), inputs/batch_NN.jsonl, manifest.json
    #   (judges write outputs/<pass>_batch_NN[_retryK].jsonl)
    $PY evaluation/llm_judge.py validate judge_<stamp> --pass pass1      # → which rows need re-requesting
    $PY evaluation/llm_judge.py retry-batch judge_<stamp> --pass pass1    # → inputs/pass1_retryK.jsonl (missing/invalid rows)
    $PY evaluation/llm_judge.py select-second judge_<stamp>               # → inputs/pass2_batch_NN.jsonl
    $PY evaluation/llm_judge.py finalize judge_<stamp>                    # → judged CSV + audit + disagreements

Blinding: judge inputs contain the letter system only — never run ids, the key, or automated ``auto_*``
hints. This module never reads the key; un-blinding happens afterwards in ``human_review.py summarize --key``.
Nothing existing is ever overwritten (every output path is new).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from human_review import COLUMNS, HUMAN_FIELDS  # noqa: E402

PROMPT_FILE = Path(__file__).resolve().parent / "judge" / "prompt_v1.md"
PROMPT_VERSION = "v1"
RUBRIC_SOURCE = Path(__file__).resolve().parent / "HUMAN_REVIEW.md"
JUDGE_FIELDS = ["review_id", "question_type", "cross_paper", "question", "gold_answer", "gold_evidence",
                "scoring_rule", "system", "answer", "retrieved_context", "citations"]
NEVER_NA = {"evidence_support", "refusal_appropriate"}
MAX_ATTEMPTS = 3          # first attempt + 2 re-requests
EVIDENCE_MAX = 400
JUDGE_MODEL = "Claude Opus (Claude Code subagent, model alias 'opus'; parent session claude-opus-5-5)"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def extract_rubric(text: str) -> str:
    """The rubric table and definitions, verbatim — without the workflow section (which names the key file)."""
    start = text.index("| Field | Scale | Definition |")
    end = text.index("\n## ", start)  # stop at the next section (AI judging / Workflow)
    return text[start:end].rstrip() + "\n"


def judge_row(row: dict) -> dict:
    return {k: row[k] for k in JUDGE_FIELDS}


def is_graph_row(row: dict) -> bool:
    return not row["citations"].startswith("n/a")


def markers_in_citations(row: dict) -> int:
    """Number of citation markers recorded for a graph row ('markers N' in the citations column)."""
    for part in row["citations"].split(";"):
        part = part.strip()
        if part.startswith("markers "):
            return int(part.split()[1])
    return 0


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists — refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    _write_new(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# --- prepare -----------------------------------------------------------------------------------------
def cmd_prepare(args) -> int:
    review = Path(args.review).resolve()
    rows = list(csv.DictReader(review.open(newline="", encoding="utf-8")))
    if any(r["run_id"] for r in rows):
        raise SystemExit("review sheet is not blinded (run_id present) — refusing to prepare judge inputs")
    if any(any(r[f].strip() for f in HUMAN_FIELDS) for r in rows):
        raise SystemExit("review sheet already contains scores — refusing (scores must not leak to judges)")
    out = review.parent / f"judge_{_now()}"
    rubric = extract_rubric(RUBRIC_SOURCE.read_text(encoding="utf-8"))
    _write_new(out / "rubric.md", rubric)
    _write_new(out / "prompt.md", PROMPT_FILE.read_text(encoding="utf-8"))
    batches = []
    for i in range(0, len(rows), args.batch_size):
        name = f"pass1_batch_{i // args.batch_size + 1:02d}"
        chunk = [judge_row(r) for r in rows[i:i + args.batch_size]]
        _write_jsonl(out / "inputs" / f"{name}.jsonl", chunk)
        batches.append({"name": name, "rows": [r["review_id"] for r in chunk],
                        "chars": sum(len(json.dumps(r)) for r in chunk)})
    key_path = review.with_name(review.stem + "_KEY.json")
    manifest = {
        "created_at": _now(), "review_file": str(review), "review_sha256": _sha256(review),
        # the key is NOT read — only its size/mtime, so later checks can show it was untouched
        "key_file": str(key_path), "key_stat": {"size": key_path.stat().st_size, "mtime": key_path.stat().st_mtime}
        if key_path.exists() else None,
        "prompt_version": PROMPT_VERSION, "prompt_sha256": _sha256(PROMPT_FILE),
        "rubric_sha256": hashlib.sha256(rubric.encode()).hexdigest(), "judge_model": JUDGE_MODEL,
        "rows": [r["review_id"] for r in rows], "batch_size": args.batch_size, "batches": batches,
        "judge_fields": JUDGE_FIELDS,
    }
    _write_new(out / "manifest.json", json.dumps(manifest, indent=2))
    total = sum(b["chars"] for b in batches)
    print(f"prepared {len(rows)} rows in {len(batches)} batches → {out}")
    print(f"input text: {total:,} chars (~{total // 4:,} tokens); largest batch {max(b['chars'] for b in batches):,} chars")
    return 0


# --- validation ------------------------------------------------------------------------------------------
def validate_judgment(j: dict, row: dict) -> list[str]:
    """Problems with one judgment for one (blinded) review row; [] if valid."""
    problems = []
    fields = j.get("fields")
    if not isinstance(fields, dict):
        return ["missing 'fields' object"]
    negative = row["question_type"] == "negative"
    cross = row["cross_paper"] == "yes"
    graph = is_graph_row(row)
    for field, allowed in HUMAN_FIELDS.items():
        f = fields.get(field)
        if not isinstance(f, dict):
            problems.append(f"{field}: missing")
            continue
        score = score_of(f)
        if score not in allowed:
            problems.append(f"{field}: score {score!r} not in {sorted(allowed)}")
        if not str(f.get("rationale", "")).strip():
            problems.append(f"{field}: empty rationale")
        if score != "NA" and not str(f.get("evidence", "")).strip():
            problems.append(f"{field}: scored without evidence")
        if not isinstance(f.get("uncertain", False), bool):
            problems.append(f"{field}: 'uncertain' must be true/false")
        # NA rules from HUMAN_REVIEW.md
        if field in ("correctness", "completeness"):
            if negative and score not in ("NA", ""):
                problems.append(f"{field}: must be NA for negative questions")
            if not negative and score == "NA":
                problems.append(f"{field}: NA only allowed for negative questions")
        if field == "citation_quality":
            if not graph and score != "NA":
                problems.append("citation_quality: must be NA for rows without citations")
            if graph and markers_in_citations(row) > 0 and score == "NA":
                problems.append("citation_quality: answer has citation markers — must be scored")
        if field == "paper_attribution" and not cross and score != "NA":
            problems.append("paper_attribution: must be NA for non-cross-paper rows")
        if field in NEVER_NA and score == "NA":
            problems.append(f"{field}: NA not allowed")
    return problems


def load_outputs(out_dir: Path, pass_name: str) -> list[tuple[str, dict]]:
    """(source file, judgment) in attempt order: batch files first, then retries in creation order."""
    files = sorted((out_dir / "outputs").glob(f"{pass_name}_*.jsonl"))
    files.sort(key=lambda p: ("_retry" in p.name, p.name))
    found = []
    for path in files:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                found.append((f"{path.name}:{n}", json.loads(line)))
            except json.JSONDecodeError as exc:
                found.append((f"{path.name}:{n}", {"__parse_error__": str(exc)}))
    return found


def collect(out_dir: Path, pass_name: str, expected: list[str], rows: dict[str, dict]) -> dict:
    """Latest valid judgment per expected row; attempts counted per row; extra/unknown ids reported."""
    attempts: dict[str, int] = {rid: 0 for rid in expected}
    valid: dict[str, dict] = {}
    problems: dict[str, list[str]] = {}
    unknown = []
    for source, j in load_outputs(out_dir, pass_name):
        rid = j.get("review_id")
        if "__parse_error__" in j or rid is None:
            unknown.append(f"{source}: unparseable or missing review_id")
            continue
        if rid not in attempts:
            unknown.append(f"{source}: unexpected review_id {rid!r}")
            continue
        attempts[rid] += 1
        errs = validate_judgment(j, rows[rid])
        if errs:
            problems[rid] = errs
        else:
            valid[rid] = {**j, "_source": source, "_attempt": attempts[rid]}
            problems.pop(rid, None)
    missing = [rid for rid in expected if rid not in valid]
    unresolved = [rid for rid in missing if attempts[rid] >= MAX_ATTEMPTS]
    return {"valid": valid, "attempts": attempts, "problems": problems, "unknown": unknown,
            "missing": missing, "retryable": [r for r in missing if r not in unresolved], "unresolved": unresolved}


def _rows(manifest: dict) -> dict[str, dict]:
    review = Path(manifest["review_file"])
    return {r["review_id"]: r for r in csv.DictReader(review.open(newline="", encoding="utf-8"))}


def scope_rows(out_dir: Path, manifest: dict) -> list[str]:
    """Rows in the judging scope: scope.json if present (a subset phase), else every row."""
    path = out_dir / "scope.json"
    return json.loads(path.read_text())["rows"] if path.exists() else manifest["rows"]


def _expected(out_dir: Path, pass_name: str, manifest: dict) -> list[str]:
    if pass_name == "pass1":
        return scope_rows(out_dir, manifest)
    sel = json.loads((out_dir / "pass2_selection.json").read_text())
    return sel["rows"]


def cmd_validate(args) -> int:
    out_dir = Path(args.judge_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    result = collect(out_dir, args.pass_name, _expected(out_dir, args.pass_name, manifest), _rows(manifest))
    print(f"{args.pass_name}: valid {len(result['valid'])}, missing {len(result['missing'])} "
          f"(retryable {len(result['retryable'])}, unresolved {len(result['unresolved'])}), "
          f"unexpected lines {len(result['unknown'])}")
    for rid, errs in list(result["problems"].items())[:20]:
        print(f"  {rid}: {'; '.join(errs)}")
    for u in result["unknown"][:10]:
        print(f"  {u}")
    return 0 if not result["missing"] else 2


def cmd_retry_batch(args) -> int:
    out_dir = Path(args.judge_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    rows = _rows(manifest)
    result = collect(out_dir, args.pass_name, _expected(out_dir, args.pass_name, manifest), rows)
    if not result["retryable"]:
        print("nothing to re-request")
        return 0
    k = 1 + len(list((out_dir / "inputs").glob(f"{args.pass_name}_retry*.jsonl")))
    path = out_dir / "inputs" / f"{args.pass_name}_retry{k}.jsonl"
    _write_jsonl(path, [judge_row(rows[rid]) for rid in result["retryable"]])
    print(f"wrote {len(result['retryable'])} rows to re-request → {path.name} (output: outputs/{path.name})")
    return 0


# --- scope (subset phase) ------------------------------------------------------------------------------
def cmd_scope(args) -> int:
    """Restrict this judging phase to question types; write pass-1 batches for scope rows that have no valid
    pass-1 judgment yet, and pass-2 batches for ALL scope rows (shuffled, so pass 2 never sees pass 1's batches)."""
    import random

    out_dir = Path(args.judge_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    rows = _rows(manifest)
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    selected = [rid for rid in manifest["rows"] if rows[rid]["question_type"] in types]
    _write_new(out_dir / "scope.json", json.dumps({"rows": selected, "types": types, "created_at": _now(),
                                                   "excluded": len(manifest["rows"]) - len(selected)}, indent=2))
    first = collect(out_dir, "pass1", selected, rows)
    todo = [rid for rid in selected if rid not in first["valid"]]
    size = args.batch_size
    for i in range(0, len(todo), size):
        _write_jsonl(out_dir / "inputs" / f"pass1_scope_batch_{i // size + 1:02d}.jsonl",
                     [judge_row(rows[r]) for r in todo[i:i + size]])
    shuffled = list(selected)
    random.Random(args.seed).shuffle(shuffled)
    _write_new(out_dir / "pass2_selection.json", json.dumps(
        {"rows": selected, "reasons": {rid: ["scope_second_pass"] for rid in selected}, "shuffle_seed": args.seed},
        indent=2))
    for i in range(0, len(shuffled), size):
        _write_jsonl(out_dir / "inputs" / f"pass2_batch_{i // size + 1:02d}.jsonl",
                     [judge_row(rows[r]) for r in shuffled[i:i + size]])
    print(f"scope {types}: {len(selected)} rows ({len(selected) - len(todo)} already judged in pass 1); "
          f"pass-1 batches for {len(todo)} rows, pass-2 batches for {len(selected)} rows (batch size {size})")
    return 0


# --- second pass ---------------------------------------------------------------------------------------
def cmd_select_second(args) -> int:
    out_dir = Path(args.judge_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    rows = _rows(manifest)
    first = collect(out_dir, "pass1", manifest["rows"], rows)
    reasons: dict[str, list[str]] = {}
    for rid in manifest["rows"]:
        if rows[rid]["cross_paper"] == "yes":
            reasons.setdefault(rid, []).append("cross_paper")
        j = first["valid"].get(rid)
        if j and any(f.get("uncertain") for f in j["fields"].values()):
            reasons.setdefault(rid, []).append("uncertain")
        if rid in first["unresolved"]:
            reasons.setdefault(rid, []).append("unresolved_in_pass1")
    selected = [rid for rid in manifest["rows"] if rid in reasons]
    _write_new(out_dir / "pass2_selection.json", json.dumps({"rows": selected, "reasons": reasons}, indent=2))
    for i in range(0, len(selected), manifest["batch_size"]):
        name = f"pass2_batch_{i // manifest['batch_size'] + 1:02d}"
        _write_jsonl(out_dir / "inputs" / f"{name}.jsonl", [judge_row(rows[r]) for r in selected[i:i + manifest["batch_size"]]])
    counts = {k: sum(k in v for v in reasons.values()) for k in ("cross_paper", "uncertain", "unresolved_in_pass1")}
    print(f"pass2: {len(selected)} rows selected {counts}")
    return 0


# --- finalize ---------------------------------------------------------------------------------------------
def score_of(field_judgment: dict) -> str:
    score = str(field_judgment.get("score", "")).strip()
    return "NA" if score.upper() == "NA" else score


def material_disagreement(field: str, a: str, b: str) -> bool:
    """Any NA/score mismatch, any change on binary fields, or a ≥ 1-point change on 0–2 fields."""
    if a == b:
        return False
    if "NA" in (a, b):
        return True
    return abs(int(a) - int(b)) >= 1


def cmd_finalize(args) -> int:
    out_dir = Path(args.judge_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    review = Path(manifest["review_file"])
    if _sha256(review) != manifest["review_sha256"]:
        raise SystemExit("original review sheet changed since prepare — refusing to finalize")
    rows = _rows(manifest)
    in_scope = set(scope_rows(out_dir, manifest))
    first = collect(out_dir, "pass1", [r for r in manifest["rows"] if r in in_scope], rows)
    second_rows = json.loads((out_dir / "pass2_selection.json").read_text())["rows"] \
        if (out_dir / "pass2_selection.json").exists() else []
    second = collect(out_dir, "pass2", second_rows, rows) if second_rows else {"valid": {}, "unresolved": [],
                                                                                 "missing": []}
    stamp = _now()
    # judged sheet: same columns as the review CSV, pass-1 scores (primary), disagreements flagged in notes
    disagreements, audit = [], []
    judged = []
    for rid in manifest["rows"]:
        row = dict(rows[rid])
        j1, j2 = first["valid"].get(rid), second["valid"].get(rid)
        flags = []
        if rid not in in_scope:
            row["notes"] = "NOT_JUDGED (outside this phase's scope)"
            judged.append(row)
            continue
        if j1:
            for field in HUMAN_FIELDS:
                row[field] = score_of(j1["fields"][field])
            if any(f.get("uncertain") for f in j1["fields"].values()):
                flags.append("uncertain:" + ",".join(k for k, f in j1["fields"].items() if f.get("uncertain")))
        else:
            flags.append("UNRESOLVED_PASS1")
        if j1 and j2:
            for field in HUMAN_FIELDS:
                a, b = score_of(j1["fields"][field]), score_of(j2["fields"][field])
                if material_disagreement(field, a, b):
                    disagreements.append({"review_id": rid, "field": field, "pass1": a, "pass2": b,
                                          "pass1_rationale": j1["fields"][field]["rationale"],
                                          "pass2_rationale": j2["fields"][field]["rationale"]})
                    flags.append(f"DISAGREE {field} {a}/{b}")
        elif rid in second_rows:
            flags.append("UNRESOLVED_PASS2")
        row["reviewer"] = f"AI judge ({PROMPT_VERSION})"
        row["notes"] = "; ".join(flags)
        judged.append(row)
        for pass_name, j in (("pass1", j1), ("pass2", j2)):
            if j:
                audit.append({"review_id": rid, "pass": pass_name, "source": j["_source"], "attempt": j["_attempt"],
                              "fields": j["fields"]})
    judged_path = out_dir / f"judged_{PROMPT_VERSION}_{stamp}.csv"
    if judged_path.exists():
        raise SystemExit(f"{judged_path} exists")
    with judged_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(judged)
    overlap = [rid for rid in second_rows if rid in first["valid"] and rid in second["valid"]]
    agreement = {}
    for field in HUMAN_FIELDS:
        pairs = [(score_of(first["valid"][r]["fields"][field]), score_of(second["valid"][r]["fields"][field]))
                 for r in overlap]
        agreement[field] = {"pairs": len(pairs), "exact_agreement": round(sum(a == b for a, b in pairs) / len(pairs), 3)
                            if pairs else None}
    meta = {**{k: manifest[k] for k in ("review_file", "review_sha256", "prompt_version", "prompt_sha256",
                                         "rubric_sha256", "judge_model")},
            "finalized_at": stamp, "label": "AI-judged (Claude subagents) — not human-verified",
            "scope": {"rows": len(in_scope), "excluded": len(manifest["rows"]) - len(in_scope),
                      "types": json.loads((out_dir / "scope.json").read_text())["types"]
                      if (out_dir / "scope.json").exists() else "all"},
            "pass1": {"valid": len(first["valid"]), "unresolved": first["unresolved"],
                      "missing": first["missing"]},
            "pass2": {"selected": len(second_rows), "valid": len(second["valid"]),
                      "unresolved": second["unresolved"], "missing": second["missing"]},
            "primary": "pass1 scores are the judged scores; pass2 only flags disagreements (no averaging)",
            "pass_independence": "both passes are Claude Opus subagents with fresh context — a self-consistency "
                                 "check, NOT independent models",
            "agreement_on_pass2_rows": agreement, "material_disagreements": len(disagreements),
            "judged_csv": judged_path.name}
    _write_new(out_dir / f"audit_{PROMPT_VERSION}_{stamp}.jsonl",
               "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audit))
    _write_new(out_dir / f"disagreements_{PROMPT_VERSION}_{stamp}.json", json.dumps(disagreements, indent=2))
    _write_new(out_dir / f"judge_run_{PROMPT_VERSION}_{stamp}.json", json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("pass1", "pass2", "material_disagreements", "agreement_on_pass2_rows",
                                          "judged_csv")}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("review")
    p.add_argument("--batch-size", type=int, default=12)
    p.set_defaults(func=cmd_prepare)
    for name, func in (("validate", cmd_validate), ("retry-batch", cmd_retry_batch)):
        p = sub.add_parser(name)
        p.add_argument("judge_dir")
        p.add_argument("--pass", dest="pass_name", default="pass1", choices=["pass1", "pass2"])
        p.set_defaults(func=func)
    p = sub.add_parser("scope")
    p.add_argument("judge_dir")
    p.add_argument("--types", default="cross_paper,negative")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=20261016)
    p.set_defaults(func=cmd_scope)
    for name, func in (("select-second", cmd_select_second), ("finalize", cmd_finalize)):
        p = sub.add_parser(name)
        p.add_argument("judge_dir")
        p.set_defaults(func=func)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
