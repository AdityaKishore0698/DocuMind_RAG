"""Stage-3 validation of the production BM25 retrieval code against the evaluation harness (local staging DB only).

Three steps, because the harness (numpy) and the API code (SQLAlchemy) live in different locked environments:

    evaluation/.venv/bin/python        evaluation/staging_bm25.py harness    --db URL --out DIR
    local_inference/.venv/bin/python   evaluation/staging_bm25.py production --db URL --out DIR [--label L]
    evaluation/.venv/bin/python        evaluation/staging_bm25.py compare    --db URL --out DIR [--production-label L]

Query embeddings are read from the staging DB's embedding cache only (no API calls; a missing one is an error).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
MODEL, DIM, USER_ID = "gemini-embedding-2", 768, 1


def _local(url: str) -> None:
    if urlparse(url).hostname not in {"localhost", "127.0.0.1", "::1"} or "staging" not in url:
        raise SystemExit("staging validation runs only on a local database whose name contains 'staging'")


def _questions() -> list[dict]:
    lines = (ROOT / "evaluation" / "questions.jsonl").read_text().splitlines()
    return [q for q in (json.loads(x) for x in lines if x.strip()) if q["status"] == "approved"]


def _pct(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


# --- step 1: harness (evaluation venv) --------------------------------------------------------------------
def cmd_harness(args) -> int:
    sys.path.insert(0, str(ROOT / "evaluation"))
    import baseline_a
    import db
    import hybrid

    conn = db.connect(args.db)
    hybrid.assert_lexical_index_fresh(conn)
    qs = _questions()
    cache = db.cached_embeddings(conn, MODEL, DIM, [q["question"] for q in qs])
    out = {}
    for q in qs:
        vec = cache.get(db.text_sha256(q["question"]))
        if vec is None:
            raise SystemExit(f"no cached embedding for {q['id']} — refusing to call the API")
        vector = db.retrieve(conn, baseline_a.RETRIEVAL_SQL, USER_ID, vec, 5)
        hyb = hybrid.hybrid_retrieve(conn, user_id=USER_ID, query_text=q["question"], query_embedding=vec, top_k=5,
                                     lexical="bm25")
        out[q["id"]] = {"vector": [r["chunk_id"] for r in vector],
                        "hybrid": [[r["chunk_id"], r["vector_rank"], r["fts_rank"], r["rrf_score"]] for r in hyb]}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "harness.json").write_text(json.dumps(out, indent=1))
    print(f"harness: {len(out)} questions")
    return 0


# --- step 2: production code path (local-inference venv: API code + SQLAlchemy) --------------------------------
def cmd_production(args) -> int:
    import os

    os.environ["DATABASE_URL"] = args.db
    sys.path.insert(0, str(ROOT / "api"))
    from sqlalchemy import text

    import models.user  # noqa: F401 — registers User for the Document relationship (as main.py does)
    import core.retrieval as R
    from core.database import SessionLocal

    os.environ.pop("RETRIEVAL_MODE", None)
    assert R.retrieval_mode() == "vector", "default must be vector"
    session = SessionLocal()
    qs = _questions()
    rows = session.execute(text("SELECT text_sha256, embedding FROM eval_embedding_cache WHERE model = :m"),
                           {"m": MODEL}).all()
    cache = {h: json.loads(e) if isinstance(e, str) else list(e) for h, e in rows}
    import hashlib

    out, lat = {}, {"vector": [], "hybrid": []}
    for q in qs:
        vec = cache.get(hashlib.sha256(q["question"].encode("utf-8")).hexdigest())
        if vec is None:
            raise SystemExit(f"no cached embedding for {q['id']} — refusing to call the API")
        entry = {}
        for mode in ("vector", "hybrid_bm25"):
            os.environ["RETRIEVAL_MODE"] = mode
            t0 = time.perf_counter()
            chunks = R.retrieve_chunks(session, user_id=USER_ID, query_text=q["question"], query_embedding=vec)
            lat["vector" if mode == "vector" else "hybrid"].append(time.perf_counter() - t0)
            entry[mode] = [c.id for c in chunks]
        os.environ["RETRIEVAL_MODE"] = "hybrid_bm25"
        entry["hybrid_rows"] = [[r.id, r.vector_rank, r.fts_rank, float(r.rrf_score)]
                                for r in R.hybrid_ranking(session, USER_ID, q["question"], vec)]
        out[q["id"]] = entry
    os.environ.pop("RETRIEVAL_MODE", None)
    result = {"questions": out, "fallbacks": R.fallback_counts(),
              "latency_ms": {m: {"p50": _pct(v, 50) * 1000, "p95": _pct(v, 95) * 1000} for m, v in lat.items()}}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / f"production_{args.label}.json"
    if path.exists():
        raise SystemExit(f"{path} exists")
    path.write_text(json.dumps(result, indent=1))
    print(f"production[{args.label}]: {len(out)} questions, fallbacks {result['fallbacks']}, latency {result['latency_ms']}")
    return 0


# --- step 3: compare + metrics (evaluation venv) -------------------------------------------------------------
def cmd_compare(args) -> int:
    sys.path.insert(0, str(ROOT / "evaluation"))
    import db
    import metrics as M

    out_dir = Path(args.out)
    harness = json.loads((out_dir / "harness.json").read_text())
    prod = json.loads((out_dir / f"production_{args.production_label}.json").read_text())
    conn = db.connect(args.db)
    qs = {q["id"]: q for q in _questions()}
    mismatches = {"vector_ids": [], "hybrid_ids": [], "hybrid_ranks_scores": []}
    for qid, h in harness.items():
        p = prod["questions"][qid]
        if p["vector"] != h["vector"]:
            mismatches["vector_ids"].append(qid)
        if p["hybrid_bm25"] != [r[0] for r in h["hybrid"]]:
            mismatches["hybrid_ids"].append(qid)
        for a, b in zip(p["hybrid_rows"], h["hybrid"]):
            if a[0] != b[0] or a[1] != b[1] or a[2] != b[2] or abs(a[3] - b[3]) > 1e-12:
                mismatches["hybrid_ranks_scores"].append(qid)
                break

    def retrieved(ids):
        with conn.cursor() as cur:
            cur.execute("SELECT c.id, c.content, p.paper_id, p.pages FROM document_chunks c JOIN eval_chunk_provenance p "
                        "ON p.chunk_id = c.id WHERE c.id = ANY(%s)", (ids,))
            by = {r[0]: r for r in cur.fetchall()}
        return [{"rank": i, "chunk_id": c, "text": by[c][1], "paper_id": by[c][2], "pages": list(by[c][3])}
                for i, c in enumerate(ids, 1)]

    per = {"vector": {}, "hybrid_bm25": {}}
    for qid, p in prod["questions"].items():
        for mode in per:
            per[mode][qid] = M.score_retrieval(qs[qid], retrieved(p[mode]), ks=(5,))

    def mean(mode, key, filt=lambda q: True):
        vals = [per[mode][i][key] for i in per[mode] if per[mode][i][key] is not None and filt(qs[i])]
        return {"mean": round(statistics.fmean(vals), 3) if vals else None, "n": len(vals)}

    split = json.loads((ROOT / "evaluation" / "splits" / "phase2_split.json").read_text())
    held = {p for g in split["heldout_groups"] for p in split["groups"][g]}
    report = {"equivalence": {k: v for k, v in mismatches.items()}, "fallbacks": prod["fallbacks"],
              "latency_ms": prod["latency_ms"], "metrics": {}}
    for mode in per:
        report["metrics"][mode] = {
            "evidence_recall@5": mean(mode, "evidence_recall@5"), "evidence_mrr": mean(mode, "evidence_mrr"),
            "page_recall@5": mean(mode, "page_recall@5"),
            "dataset_recall@5": mean(mode, "evidence_recall@5", lambda q: q["type"] == "dataset"),
            "heldout_recall@5": mean(mode, "evidence_recall@5", lambda q: q.get("papers") and set(q["papers"]) <= held),
            "heldout_mrr": mean(mode, "evidence_mrr", lambda q: q.get("papers") and set(q["papers"]) <= held)}
    path = out_dir / f"compare_{args.production_label}.json"
    if path.exists():
        raise SystemExit(f"{path} exists")
    path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["harness", "production", "compare"])
    parser.add_argument("--db", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="run1")
    parser.add_argument("--production-label", default="run1")
    args = parser.parse_args()
    _local(args.db)
    return {"harness": cmd_harness, "production": cmd_production, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
