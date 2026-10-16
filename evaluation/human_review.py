"""Consolidated human review across systems (see evaluation/HUMAN_REVIEW.md).

    PY=evaluation/.venv/bin/python
    $PY evaluation/human_review.py build --run L=20261015T161715Z_L --run BL=20261015T165903Z_BL \\
        --run CL=20261015T171852Z_CL --run CLV=20261015T175700Z_CLV [--blind]
    $PY evaluation/human_review.py summarize evaluation/results/human_review/<file>.csv
    $PY evaluation/human_review.py export-run-scores evaluation/results/human_review/<file>.csv

* build      — one row per (question, system) from each run's FINAL trace per question; never overwrites.
               --blind replaces system names by per-question letters and writes the key to a separate file.
* summarize  — validates the human fields and writes a timestamped summary (JSON + Markdown).
* export-run-scores — writes each run's human_scores.csv (correctness / paper_attribution) so
               `run_eval.py report <RUN_ID>` includes human correctness; refuses to overwrite.

Nothing here assigns scores: automated columns are labelled ``auto_*`` and are context, not grades.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as M  # noqa: E402
from common import QUESTIONS_FILE, RUNS_DIR, load_questions  # noqa: E402

REVIEW_DIR = RUNS_DIR.parent / "human_review"
EXCERPT_CHARS = 400
# field -> allowed values ("" = not yet scored; NA only where the guide allows it)
HUMAN_FIELDS = {
    "correctness": {"0", "1", "2", "NA"},
    "evidence_support": {"0", "1", "2"},
    "completeness": {"0", "1", "2", "NA"},
    "citation_quality": {"0", "1", "2", "NA"},
    "refusal_appropriate": {"0", "1"},
    "paper_attribution": {"0", "1", "NA"},
}
SCALES = {"correctness": 2, "evidence_support": 2, "completeness": 2, "citation_quality": 2,
          "refusal_appropriate": 1, "paper_attribution": 1}
COLUMNS = (["review_id", "question_id", "question_type", "cross_paper", "question", "gold_answer", "gold_evidence",
            "scoring_rule", "system", "run_id", "answer", "retrieved_context", "citations",
            "auto_gold_quote_in_context", "auto_citation_validity"]
           + list(HUMAN_FIELDS) + ["reviewer", "notes"])


def latest_traces(run_id: str) -> dict[str, dict]:
    path = RUNS_DIR / run_id / "traces.jsonl"
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            latest[t["question_id"]] = t
    return latest


def _context_text(retrieved: list[dict]) -> str:
    parts = []
    for i, c in enumerate(retrieved or [], start=1):
        label = c.get("source_label") or c.get("paper_id") or "?"
        text = " ".join((c.get("text") or "").split())[:EXCERPT_CHARS]
        parts.append(f"[S{i}] {label} (paper {c.get('paper_id')}, pages {c.get('pages')}): {text}")
    return "\n".join(parts)


def _gold_text(q: dict) -> str:
    return "\n".join(f"- {e['paper']} p.{e['page']}: \"{e['quote']}\"" for e in q.get("evidence", []))


def _citations_text(trace: dict) -> str:
    graph = trace.get("graph")
    if not graph:
        return "n/a (system is not prompted to cite)"
    c = graph["citations"]
    return (f"cited sources {c['valid_sources']}; invalid markers {c['invalid_sources']}; "
            f"markers {c['markers']}; cited chunk ids {c.get('cited_chunk_ids')}")


def build_rows(runs: dict[str, str], questions: list[dict]) -> list[dict]:
    traces = {system: latest_traces(run_id) for system, run_id in runs.items()}
    rows = []
    for q in questions:
        for system, run_id in runs.items():
            t = traces[system].get(q["id"])
            if t is None:
                raise SystemExit(f"run {run_id} has no trace for {q['id']} — finish the run first")
            if t.get("outcome") not in (None, "answered"):
                raise SystemExit(f"run {run_id}: {q['id']} outcome is {t.get('outcome')} — not reviewable yet")
            retrieval = M.score_retrieval(q, t.get("retrieved") or [], ks=(5,))
            cites = M.score_citations(q, t["graph"]["citations"], t["retrieved"]) if t.get("graph") else {}
            rows.append({
                "review_id": f"{q['id']}|{system}", "question_id": q["id"], "question_type": q["type"],
                "cross_paper": "yes" if q["type"] == "cross_paper" else "no", "question": q["question"],
                "gold_answer": q.get("gold_answer", ""), "gold_evidence": _gold_text(q),
                "scoring_rule": q.get("scoring_rule", ""), "system": system, "run_id": run_id,
                "answer": t.get("answer", ""), "retrieved_context": _context_text(t.get("retrieved")),
                "citations": _citations_text(t),
                "auto_gold_quote_in_context": "" if retrieval.get("evidence_recall@5") is None
                else f"{retrieval['evidence_recall@5']:.2f}",
                "auto_citation_validity": "" if cites.get("citation_validity") is None
                else f"{cites['citation_validity']:.2f}",
                **{f: "" for f in HUMAN_FIELDS}, "reviewer": "", "notes": "",
            })
    return rows


def blind(rows: list[dict], seed: int) -> tuple[list[dict], dict]:
    """Shuffle systems within each question and replace names by letters; returns (rows, key)."""
    rng = random.Random(seed)
    out, key = [], {}
    by_q: dict[str, list[dict]] = {}
    for r in rows:
        by_q.setdefault(r["question_id"], []).append(r)
    for qid, group in by_q.items():
        rng.shuffle(group)
        for letter, r in zip("ABCDEFGH", group):
            key[f"{qid}|{letter}"] = {"system": r["system"], "run_id": r["run_id"]}
            out.append({**r, "review_id": f"{qid}|{letter}", "system": letter, "run_id": ""})
    return out, key


def _write_new(path: Path, write) -> None:
    if path.exists():
        raise SystemExit(f"{path} already exists — refusing to overwrite (human annotations may be in it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path)


def cmd_build(args) -> int:
    runs = dict(item.split("=", 1) for item in args.run)
    questions = [q for q in load_questions() if q["status"] == "approved"]
    if args.cross_paper_first:
        questions.sort(key=lambda q: q["type"] != "cross_paper")
    rows = build_rows(runs, questions)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(args.out) if args.out else REVIEW_DIR / f"review_{'-'.join(runs)}_{stamp}{'_blind' if args.blind else ''}.csv"
    key = None
    if args.blind:
        rows, key = blind(rows, args.seed)

    def write(p):
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)

    _write_new(path, write)
    meta = {"runs": runs, "questions_file": str(QUESTIONS_FILE), "rows": len(rows), "blind": args.blind,
            "created_at": stamp}
    if key is not None:
        meta["seed"] = args.seed
        key_path = path.with_name(path.stem + "_KEY.json")
        _write_new(key_path, lambda p: p.write_text(json.dumps({"meta": meta, "key": key}, indent=2)))
        print(f"blind key (do not open while scoring): {key_path}")
    print(f"wrote {len(rows)} rows ({len(questions)} questions × {len(runs)} systems): {path}")
    return 0


def read_scored(path: Path, key_path: Path | None = None) -> list[dict]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    key_path = key_path or path.with_name(path.stem + "_KEY.json")
    key = json.loads(key_path.read_text())["key"] if key_path.exists() else None
    errors = []
    for i, r in enumerate(rows, start=2):
        if key is not None:
            real = key.get(r["review_id"])
            if real is None:
                errors.append(f"line {i}: review_id {r['review_id']} not in blind key")
                continue
            r["system"], r["run_id"] = real["system"], real["run_id"]
        for field, allowed in HUMAN_FIELDS.items():
            value = (r.get(field) or "").strip()
            if value.upper() == "NA":
                value = "NA"
            if value and value not in allowed:
                errors.append(f"line {i} ({r['review_id']}): {field}={value!r} not in {sorted(allowed)}")
            r[field] = value
    if errors:
        raise SystemExit("invalid human scores:\n  " + "\n  ".join(errors[:30]))
    return rows


def _mean(values: list[str], scale: int) -> dict:
    nums = [float(v) / scale for v in values if v not in ("", "NA")]
    return {"mean": round(sum(nums) / len(nums), 3) if nums else None, "n": len(nums)}


def _sign_test(pairs: list[tuple[float, float]]) -> dict:
    better = sum(a > b for a, b in pairs)
    worse = sum(a < b for a, b in pairs)
    n = better + worse
    p = min(1.0, 2 * sum(comb(n, i) for i in range(max(better, worse), n + 1)) / 2 ** n) if n else 1.0
    return {"better": better, "worse": worse, "tied": len(pairs) - n, "sign_test_p": round(p, 3)}


def summarize(rows: list[dict]) -> dict:
    systems = list(dict.fromkeys(r["system"] for r in rows))
    scored = [r for r in rows if any(r[f] for f in HUMAN_FIELDS)]
    out = {"rows": len(rows), "rows_with_any_score": len(scored), "systems": {}, "cross_paper": {}, "by_type": {},
           "paired_correctness": {}, "note": "means are normalised to 0..1 (score / max of scale); n = scored rows"}
    for s in systems:
        mine = [r for r in rows if r["system"] == s]
        out["systems"][s] = {f: _mean([r[f] for r in mine], SCALES[f]) for f in HUMAN_FIELDS}
        out["cross_paper"][s] = {f: _mean([r[f] for r in mine if r["cross_paper"] == "yes"], SCALES[f])
                                 for f in ("correctness", "completeness", "paper_attribution", "evidence_support")}
        for qtype in sorted({r["question_type"] for r in mine}):
            out["by_type"].setdefault(qtype, {})[s] = _mean(
                [r["correctness"] for r in mine if r["question_type"] == qtype], 2)
    corr = {(r["question_id"], r["system"]): r["correctness"] for r in rows}
    for a in systems:
        for b in systems:
            if a < b:
                pairs = [(float(corr[(q, a)]), float(corr[(q, b)])) for (q, s) in corr if s == a
                         and corr.get((q, b), "") not in ("", "NA") and corr[(q, a)] not in ("", "NA")]
                out["paired_correctness"][f"{a} vs {b}"] = _sign_test(pairs)
    return out


def cmd_summarize(args) -> int:
    path = Path(args.file)
    summary = summarize(read_scored(path, Path(args.key) if args.key else None))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = path.with_name(f"{path.stem}_summary_{stamp}.json")
    _write_new(json_path, lambda p: p.write_text(json.dumps(summary, indent=2)))
    title = "AI-judged summary (not human-verified)" if path.name.startswith("judged_") else "Human review summary"
    lines = [f"# {title} — {path.name}", "",
             f"Scored rows: {summary['rows_with_any_score']} / {summary['rows']} (means normalised to 0–1)", "",
             "| system | " + " | ".join(HUMAN_FIELDS) + " |", "|---" * (len(HUMAN_FIELDS) + 1) + "|"]
    for s, v in summary["systems"].items():
        lines.append(f"| {s} | " + " | ".join("n/a" if v[f]["mean"] is None else f"{v[f]['mean']:.2f} (n={v[f]['n']})"
                                              for f in HUMAN_FIELDS) + " |")
    lines += ["", "Cross-paper questions (correctness / completeness / paper_attribution):", ""]
    def cell(stat: dict) -> str:
        return "n/a" if stat["mean"] is None else f"{stat['mean']:.2f}"

    for s, v in summary["cross_paper"].items():
        lines.append(f"- {s}: " + ", ".join(f"{f} {cell(v[f])} (n={v[f]['n']})"
                                            for f in ("correctness", "completeness", "paper_attribution")))
    lines += ["", "Paired correctness (per question, sign test):", ""]
    lines += [f"- {k}: {v}" for k, v in summary["paired_correctness"].items()]
    md_path = path.with_name(f"{path.stem}_summary_{stamp}.md")
    _write_new(md_path, lambda p: p.write_text("\n".join(lines) + "\n"))
    print("\n".join(lines))
    print(f"\nwrote {json_path.name} and {md_path.name}")
    return 0


def cmd_export_run_scores(args) -> int:
    """Per-run human_scores.csv (0/1/2 correctness, 0/1 paper_attribution) for `run_eval.py report`."""
    rows = read_scored(Path(args.file))
    by_run: dict[str, list[dict]] = {}
    for r in rows:
        if r["correctness"] not in ("", "NA"):
            by_run.setdefault(r["run_id"], []).append(r)
    for run_id, items in by_run.items():
        target = RUNS_DIR / run_id / "human_scores.csv"

        def write(p, items=items):
            with p.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["question_id", "correctness", "paper_attribution", "notes"])
                w.writeheader()
                for r in items:
                    w.writerow({"question_id": r["question_id"], "correctness": r["correctness"],
                                "paper_attribution": "" if r["paper_attribution"] == "NA" else r["paper_attribution"],
                                "notes": r["notes"]})

        _write_new(target, write)
        print(f"wrote {target} ({len(items)} scored questions) — now: run_eval.py report {run_id}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--run", action="append", required=True, metavar="SYSTEM=RUN_ID")
    p.add_argument("--blind", action="store_true", help="hide system names (key written separately)")
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--cross-paper-first", action="store_true", help="put the cross-paper questions first")
    p.add_argument("--out")
    p.set_defaults(func=cmd_build)
    p = sub.add_parser("summarize")
    p.add_argument("file")
    p.add_argument("--key", help="blind key of the sheet the scores came from (default: <file stem>_KEY.json)")
    p.set_defaults(func=cmd_summarize)
    p = sub.add_parser("export-run-scores")
    p.add_argument("file")
    p.set_defaults(func=cmd_export_run_scores)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
