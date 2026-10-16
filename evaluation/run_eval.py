"""DocuMind evaluation harness CLI.  See evaluation/README.md for the full workflow.

Offline (no Gemini calls):
    run_eval.py init-db                       # local pgvector DB: production schema + eval tables
    run_eval.py extract                       # per-page text from the downloaded PDFs
    run_eval.py validate-annotations          # schema + every gold quote found on its page
    run_eval.py estimate                      # predicted Gemini requests/characters

Gemini calls (require GEMINI_API_KEY in the environment):
    run_eval.py check-embedding-equivalence   # batched vs per-chunk vectors (1 + N requests)
    run_eval.py ingest --system A             # chunk + embed + store the corpus
    run_eval.py run --system A                # answer the approved questions
    run_eval.py run --resume RUN_ID           # continue a stopped run (e.g. after a daily quota)
Offline again:
    run_eval.py report RUN_ID                 # metrics, report.md, human-review sheet
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Callable

import baseline_a
import metrics as M
from common import (
    PAGES_DIR, QUESTIONS_FILE, RUNS_DIR, SUMMARIES_DIR, load_corpus, load_pages, load_questions,
    setup_logging, validate_questions,
)
import preprocess
from provenance import concat_with_spans, map_chunks, page_texts_from_pdf

log = setup_logging()


# --- systems under test ---------------------------------------------------------
@dataclass(frozen=True)
class System:
    name: str
    description: str
    embedding_model: str
    embedding_dim: int
    chat_model: str
    top_k: int
    retrieval_sql: str
    extract_text: Callable[[bytes, str], str]
    split_text: Callable[[str], list[str]]
    build_system_prompt: Callable[[str], str]
    build_context: Callable[[list[str]], str]
    preprocess: Callable[[str], str]  # evaluation-only text preprocessing applied after extraction
    deviations: tuple[str, ...]       # recorded in every ingest/run output
    provider: str = "gemini"          # "gemini" (gemini.py) or "ollama" (ollama_backend.py)
    retrieval: str = "vector"         # "vector" (the system's SQL) or "hybrid_bm25" (hybrid.py)
    mode: str = "oneshot"             # "oneshot" (retrieve → prompt → answer) or "graph" (agentic.py)
    corpus_system: str = ""           # read chunks ingested by another system (same embedding space)

    @property
    def corpus(self) -> str:
        return self.corpus_system or self.name


SYSTEMS: dict[str, System] = {
    "A": System(
        name="A",
        description=(f"Production one-shot RAG, frozen at {baseline_a.BASELINE_COMMIT[:7]} "
                     "(+ evaluation-only NUL stripping; see preprocess.py)"),
        embedding_model=baseline_a.EMBEDDING_MODEL,
        embedding_dim=baseline_a.EMBEDDING_DIM,
        chat_model=baseline_a.CHAT_MODEL,
        top_k=baseline_a.TOP_K,
        retrieval_sql=baseline_a.RETRIEVAL_SQL,
        extract_text=baseline_a.extract_text,
        split_text=baseline_a.split_text,
        build_system_prompt=baseline_a.build_system_prompt,
        build_context=baseline_a.build_context,
        preprocess=preprocess.strip_nul,
        deviations=(preprocess.DEVIATION,
                    "batched ingestion embeddings (verified equivalent to per-chunk calls)",
                    "harness-side page provenance for scoring only"),
    ),
    # Local-model baseline: System A's frozen pipeline (extraction, NUL stripping, chunking, retrieval
    # SQL, prompt, top-k) with ONLY the two model calls swapped for local Ollama models. It is not
    # Baseline A: both the embedding space and the generator differ, so results are not interchangeable.
    "L": System(
        name="L",
        description=("Local-model baseline — System A pipeline with qwen3:8b generation and nomic-embed-text "
                     "embeddings via Ollama (NOT the Gemini Baseline A)"),
        embedding_model="nomic-embed-text",
        embedding_dim=768,
        chat_model="qwen3:8b",
        top_k=baseline_a.TOP_K,
        retrieval_sql=baseline_a.RETRIEVAL_SQL,
        extract_text=baseline_a.extract_text,
        split_text=baseline_a.split_text,
        build_system_prompt=baseline_a.build_system_prompt,
        build_context=baseline_a.build_context,
        preprocess=preprocess.strip_nul,
        deviations=(preprocess.DEVIATION,
                    "embedding model nomic-embed-text (768-d) with search_document:/search_query: prefixes",
                    "generation qwen3:8b via Ollama, think=false, num_ctx 8192, model-default sampling",
                    "batched ingestion embeddings via /api/embed",
                    "harness-side page provenance for scoring only"),
        provider="ollama",
    ),
}

# Controlled Phase 2/3 variants on System L's chunks and models (qwen3:8b + nomic-embed-text):
# only the retrieval method and/or the orchestration differ from L.
_L = SYSTEMS["L"]
SYSTEMS["BL"] = replace(
    _L, name="BL", corpus_system="L", retrieval="hybrid_bm25",
    description="System L with BM25 hybrid retrieval (vector + PostgreSQL FTS + RRF), one-shot answer",
    deviations=_L.deviations + ("retrieval: BM25 hybrid (k1 1.2, b 0.75, RRF k 60, 20 candidates/list)",))
SYSTEMS["CL"] = replace(
    _L, name="CL", corpus_system="L", retrieval="hybrid_bm25", mode="graph",
    description="Bounded LangGraph workflow (plan → retrieve → assess → synthesize → verify) on System L's "
                "models and chunks, BM25 hybrid retrieval, cited answer from a 5-chunk context",
    deviations=_L.deviations + ("retrieval: BM25 hybrid", "orchestration: agentic.py graph (≤3 subqueries, ≤2 "
                                "retrieval rounds); synthesis prompt asks for [S#] citations with source filenames"))
SYSTEMS["CLV"] = replace(
    SYSTEMS["CL"], name="CLV", retrieval="vector",
    description="Same LangGraph workflow as CL but vector-only retrieval (isolates the orchestration effect)",
    deviations=_L.deviations + ("retrieval: vector", "orchestration: agentic.py graph (≤3 subqueries, ≤2 "
                                "retrieval rounds); synthesis prompt asks for [S#] citations with source filenames"))
# "B"/"C" on Gemini follow once the Gemini quota allows comparable runs.

TRACKED_PACKAGES = ("PyPDF2", "langchain-text-splitters", "langchain-core", "google-genai",
                    "psycopg2-binary", "pgvector", "numpy", "httpx")


def make_embedder(system: System, *, stats, conn, embed_mode: str = "batch"):
    """The system's embedder (Gemini or Ollama) behind one interface."""
    if system.provider == "ollama":
        from ollama_backend import OllamaEmbedder

        return OllamaEmbedder(model=system.embedding_model, dim=system.embedding_dim, stats=stats, conn=conn,
                              batch_size=env_int("EVAL_OLLAMA_EMBED_BATCH", 32),
                              max_retries=env_int("EVAL_MAX_RETRIES", 2))
    from gemini import Embedder, make_client

    return Embedder(make_client(), model=system.embedding_model, dim=system.embedding_dim, stats=stats,
                    conn=conn, batch_size=env_int("EVAL_EMBED_BATCH_SIZE", 45), mode=embed_mode,
                    items_per_min=env_int("EVAL_EMBED_ITEMS_PER_MIN", 90),
                    max_retries=env_int("EVAL_MAX_RETRIES", 4))


def assert_single_embedding_space(conn, system: System) -> None:
    """One system (= one embedding space) per evaluation database."""
    import db

    others = db.systems_present(conn) - {system.corpus}
    if others:
        raise SystemExit(f"this evaluation DB already holds vectors of system(s) {sorted(others)}; System "
                         f"{system.name} ({system.embedding_model}) needs its own database — refusing to mix "
                         f"embedding spaces")


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def environment_metadata() -> dict:
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "git_head": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "baseline_commit": baseline_a.BASELINE_COMMIT,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


# --- offline commands ----------------------------------------------------------------
def cmd_init_db(args) -> int:
    import db

    url = db.eval_database_url()
    db.assert_local_eval_database(url)
    conn = db.connect(url)
    db.init_schema(conn)
    user_id = db.ensure_eval_user(conn)
    log.info("initialised %s — production schema from scripts/init_db.sql + eval tables; eval user id=%s",
             db.redacted(url), user_id)
    log.info("server: %s", db.corpus_stats(conn))
    return 0


def cmd_extract(args) -> int:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for paper in load_corpus():
        if not paper.pdf_path.exists():
            log.error("%s: PDF missing — run download_corpus.py first", paper.id)
            failures += 1
            continue
        data = paper.pdf_path.read_bytes()
        pages = page_texts_from_pdf(data)
        concatenated, spans = concat_with_spans(pages)
        production_text = baseline_a.extract_text(data, paper.filename)
        if concatenated != production_text:
            log.error("%s: page reconstruction does not match extract_text output", paper.id)
            failures += 1
            continue
        empty = [i for i, t in enumerate(pages, start=1) if not t.strip()]
        paper.pages_path.write_text(json.dumps({
            "paper_id": paper.id, "pdf_sha256": _sha256(paper.pdf_path), "num_pages": len(pages),
            "empty_pages": empty, "chars": len(production_text), "pages": pages,
        }))
        log.info("%-18s %3d pages  %7d chars  empty=%s", paper.id, len(pages), len(production_text), empty or "-")
    return 1 if failures else 0


def cmd_validate(args) -> int:
    corpus = {p.id: p for p in load_corpus(enabled_only=False)}
    questions = load_questions()
    errors = validate_questions(questions, set(corpus))
    warnings = []
    enabled = {p.id for p in load_corpus()}
    page_cache: dict[str, list[str]] = {}
    for q in questions:
        for i, ev in enumerate(q.get("evidence", [])):
            pid = ev.get("paper")
            if pid not in enabled:
                warnings.append(f"{q['id']}: evidence paper {pid!r} is disabled in corpus.yaml")
                continue
            if pid not in page_cache:
                if not corpus[pid].pages_path.exists():
                    errors.append(f"{q['id']}: no extracted pages for {pid} — run `extract`")
                    page_cache[pid] = []
                    continue
                page_cache[pid] = load_pages(corpus[pid])
            pages = page_cache[pid]
            if not pages:
                continue
            for j, variant in enumerate(M.variants(ev)):
                label = f"evidence[{i}]" if j == 0 else f"evidence[{i}].alternatives[{j - 1}]"
                page = variant.get("page", 0)
                if not isinstance(page, int) or not 1 <= page <= len(pages):
                    errors.append(f"{q['id']}: {label} page {page} outside 1..{len(pages)}")
                    continue
                quote = M.normalize(variant.get("quote", ""))
                if quote not in M.normalize(pages[page - 1]):
                    found = [n for n, t in enumerate(pages, start=1) if quote in M.normalize(t)]
                    hint = f" (found on page(s) {found})" if found else ""
                    errors.append(f"{q['id']}: {label} quote not on page {page}{hint}")
    for w in warnings:
        log.warning(w)
    for e in errors:
        log.error(e)
    counts = {}
    for q in questions:
        counts[q["type"]] = counts.get(q["type"], 0) + 1
    statuses = {s: sum(q.get("status") == s for q in questions) for s in ("draft", "approved", "rejected")}
    log.info("%d questions %s | status %s | %d errors, %d warnings",
             len(questions), counts, statuses, len(errors), len(warnings))
    return 1 if errors else 0


def cmd_estimate(args) -> int:
    system = SYSTEMS[args.system]
    batch_size = env_int("EVAL_EMBED_BATCH_SIZE", 50)
    total_chunks = total_chars = batch_requests = 0
    for paper in load_corpus():
        if not paper.pages_path.exists():
            log.error("%s: run `extract` first", paper.id)
            return 1
        text, _ = concat_with_spans(load_pages(paper), system.preprocess)
        chunks = system.split_text(text)
        chars = sum(len(c) for c in chunks)
        total_chunks += len(chunks)
        total_chars += chars
        batch_requests += math.ceil(len(chunks) / batch_size)
        log.info("%-18s %5d chunks  %8d chars", paper.id, len(chunks), chars)
    questions = [q for q in load_questions() if args.include_drafts or q["status"] == "approved"]
    prompt_chars = len(system.build_system_prompt("")) + system.top_k * 1000
    log.info("--- System %s estimate (no API calls made) ---", system.name)
    log.info("ingestion: %d chunks, %d chars (~%d tokens at 4 chars/token, heuristic)",
             total_chunks, total_chars, total_chars // 4)
    log.info("  embedding requests: %d batched (batch size %d) | %d if per-chunk (production pattern)",
             batch_requests, batch_size, total_chunks)
    log.info("questions selected: %d (%s)", len(questions), "incl. drafts" if args.include_drafts else "approved only")
    log.info("  per run: %d query-embedding requests + %d generation requests", len(questions), len(questions))
    log.info("  generation input per question: <= ~%d chars (~%d tokens) + question; output tokens unknown "
             "(gemini-3.6-flash is a thinking model)", prompt_chars, prompt_chars // 4)
    return 0


# --- Gemini commands --------------------------------------------------------------
def cmd_check_equivalence(args) -> int:
    import numpy as np

    from gemini import CallStats, Embedder, make_client

    if SYSTEMS[args.system].provider != "gemini":
        log.error("check-embedding-equivalence applies to the Gemini batching path only")
        return 1

    system = SYSTEMS[args.system]
    paper = load_corpus()[0]
    text, _ = concat_with_spans(load_pages(paper), system.preprocess)
    sample = system.split_text(text)[: args.samples]
    stats = CallStats()
    embedder = Embedder(make_client(), model=system.embedding_model, dim=system.embedding_dim,
                        stats=stats, conn=None, batch_size=len(sample))
    batched = embedder._embed_batch(sample)
    single = [embedder.embed_one(t) for t in sample]
    sims = []
    for a, b in zip(batched, single):
        a, b = np.asarray(a), np.asarray(b)
        sims.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    ok = len(batched) == len(sample) and stats.batch_fallbacks == 0 and min(sims) >= 0.9999
    log.info("samples=%d  batch_fallbacks=%d  cosine(batched, single): min=%.6f max=%.6f  -> %s",
             len(sample), stats.batch_fallbacks, min(sims), max(sims), "EQUIVALENT" if ok else "NOT EQUIVALENT")
    log.info("requests: %s", stats.requests)
    return 0 if ok else 1


def cmd_ingest(args) -> int:
    import db
    from gemini import CallStats

    from gemini import RetryBudgetExhausted

    system = SYSTEMS[args.system]
    if system.corpus != system.name:
        log.error("System %s reads System %s's chunks — run `ingest --system %s` instead",
                  system.name, system.corpus, system.corpus)
        return 1
    conn = db.connect()
    assert_single_embedding_space(conn, system)
    user_id = db.ensure_eval_user(conn)
    stats = CallStats()
    started = time.perf_counter()
    report: list[dict] = []
    outcome = {"status": "running", "error": None, "remaining_papers": []}
    papers = [p for p in load_corpus() if not args.papers or p.id in set(args.papers)]
    path = RUNS_DIR / f"ingest_{system.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    try:
        # 1. Prepare every paper's chunks up front (offline) so the state check
        #    can compare stored chunk counts with what the chunker produces now.
        prepared = {}
        nul_removed: dict[str, int] = {}
        outcome["preprocessing"] = {"deviations": list(system.deviations), "nul_removed_by_paper": nul_removed}
        for paper in papers:
            data = paper.pdf_path.read_bytes()
            raw_text = system.extract_text(data, paper.filename)           # frozen production extraction
            text = system.preprocess(raw_text)                             # evaluation-only deviation
            page_text, spans = concat_with_spans(page_texts_from_pdf(data), system.preprocess)
            if page_text != text:
                raise RuntimeError(f"{paper.id}: provenance text differs from extract_text output")
            chunks = system.split_text(text)
            removed = preprocess.nul_count(raw_text)
            if removed:
                log.info("%-18s preprocessing removed %d NUL character(s)", paper.id, removed)
            nul_removed[paper.id] = removed
            prepared[paper.id] = (_sha256(paper.pdf_path), chunks, map_chunks(text, spans, chunks))

        # 2. Consistency check before touching anything.
        state = db.ingest_state_issues(conn, system.name, {pid: len(v[1]) for pid, v in prepared.items()})
        log.info("existing state: %s", state["stored"] or "empty")
        if state["blocking"]:
            for issue in state["blocking"]:
                log.error("inconsistent eval DB: %s", issue)
            raise RuntimeError("eval DB has rows the harness cannot attribute — not deleting them automatically")
        for pid in state["reingest"]:
            log.warning("%s: stored chunk count differs from the chunker — deleting it for re-ingestion", pid)
            db.delete_paper(conn, system.name, pid)

        embedder = make_embedder(system, stats=stats, conn=conn, embed_mode=args.embed_mode)
        log.info("embedding pacing: %d items/min, batch size %d, max retries %d",
                 embedder.pacer.limit, embedder.batch_size, embedder.max_retries)

        # 3. Ingest what is missing; a paper is inserted in ONE transaction only
        #    after all its embeddings exist, so a failure leaves no partial paper
        #    (finished batches stay in the embedding cache for the next attempt).
        outcome["remaining_papers"] = [p.id for p in papers]
        for paper in papers:
            pdf_sha, chunks, provenance = prepared[paper.id]
            if db.paper_already_ingested(conn, system.name, paper.id, pdf_sha) and not args.reingest:
                log.info("%-18s already ingested (%d chunks) — reusing", paper.id, len(chunks))
                outcome["remaining_papers"].remove(paper.id)
                report.append({"paper": paper.id, "chunks": len(chunks), "status": "reused"})
                continue
            db.delete_paper(conn, system.name, paper.id)
            cached_before = stats.embed_items_from_cache
            t0 = time.perf_counter()
            vectors = embedder.embed_many(chunks)
            embed_s = time.perf_counter() - t0
            db.insert_document_with_chunks(conn, system=system.name, user_id=user_id, paper=paper,
                                           pdf_sha256=pdf_sha, provenance=provenance, vectors=vectors)
            unmapped = sum(not p.mapped for p in provenance)
            report.append({"paper": paper.id, "chunks": len(chunks), "unmapped_chunks": unmapped,
                           "from_cache": stats.embed_items_from_cache - cached_before,
                           "embed_s": round(embed_s, 2), "status": "ingested"})
            outcome["remaining_papers"].remove(paper.id)
            log.info("%-18s %4d chunks (%d unmapped, %d from cache)  embed %.1fs", paper.id, len(chunks), unmapped,
                     stats.embed_items_from_cache - cached_before, embed_s)
        outcome["status"] = "completed"
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('eval_bm25_stats')")
            has_lexical = cur.fetchone()[0] is not None
        if has_lexical and any(r["status"] == "ingested" for r in report):
            import hybrid

            outcome["lexical_index"] = hybrid.build_lexical_index(conn)  # keep BM25 stats in step with the corpus
            log.info("BM25 statistics rebuilt after ingestion")
        return 0
    except RetryBudgetExhausted as exc:
        outcome["status"] = "stopped_retry_budget_exhausted"
        outcome["error"] = {"kind": exc.kind, "category": exc.category, "attempts": exc.attempts,
                            **exc.details, "quota": exc.quota, "retry_after_s": exc.retry_after_s}
        log.error("STOPPED: %s (HTTP %s %s: %s) — quota %s; remaining papers: %s", exc,
                  exc.details["http_status"], exc.details["status"], exc.details["message"],
                  exc.quota, outcome["remaining_papers"])
        return 2
    except Exception as exc:  # noqa: BLE001 — persisted below, then reported
        outcome["status"] = "failed"
        outcome["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        log.error("ingestion failed: %s: %s", type(exc).__name__, str(exc)[:300])
        return 1
    finally:
        out = {"system": system.name, "provider": system.provider, "embedding_model": system.embedding_model,
               "embed_mode": args.embed_mode, **outcome,
               "elapsed_s": round(time.perf_counter() - started, 2), "papers": report, "stats": stats.as_dict(),
               "db": db.corpus_stats(conn), "environment": environment_metadata()}
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, default=str))
        log.info("ingest %s | requests %s | attempts %s | items embedded %d (attempted %d, cache %d) | retries %d "
                 "(%.0fs waiting) | pacing %.0fs | final failures %s | wrote %s",
                 outcome["status"], stats.requests, stats.attempts, stats.embed_items_embedded,
                 stats.embed_items_attempted, stats.embed_items_from_cache, stats.retries, stats.retry_sleep_s,
                 stats.pacing_sleep_s, stats.final_failures or "none", path.relative_to(RUNS_DIR.parent.parent))


def _model_metadata(client, models: list[str]) -> dict:
    out = {}
    for name in models:
        try:
            m = client.models.get(model=name)
            out[name] = {"version": getattr(m, "version", None), "display_name": getattr(m, "display_name", None)}
        except Exception as exc:  # noqa: BLE001 — metadata is informational
            out[name] = {"error": type(exc).__name__}
    return out


# --- run (resumable) -----------------------------------------------------------------
# traces.jsonl is an append-only log: one record per attempt at a question. A question's
# state is its LATEST record. Nothing is ever rewritten; a resume only appends.
ANSWERED, INCOMPLETE, FAILED_RETRYABLE, FAILED_TERMINAL = "answered", "incomplete", "failed_retryable", "failed_terminal"
TERMINAL_OUTCOMES = {ANSWERED, FAILED_TERMINAL}
RETRYABLE_CATEGORIES = {"rate_limit", "server", "timeout", "daily_quota", "harness_or_embedding"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def trace_outcome(trace: dict) -> str:
    """Outcome of one trace record (older records without an ``outcome`` field are derived)."""
    if trace.get("outcome"):
        return trace["outcome"]
    error = trace.get("error")
    if not error:
        return ANSWERED if trace.get("retrieved") is not None else FAILED_RETRYABLE
    if trace.get("answer"):
        return INCOMPLETE  # text streamed before the failure: a partial answer is never a success
    return FAILED_RETRYABLE if error.get("category") in RETRYABLE_CATEGORIES else FAILED_TERMINAL


def read_traces(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latest_traces(records: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for record in records:
        latest[record["question_id"]] = record
    return latest


def pending_question_ids(question_ids: list[str], records: list[dict]) -> list[str]:
    """Questions without a terminal outcome, in benchmark order."""
    latest = latest_traces(records)
    return [qid for qid in question_ids if qid not in latest or trace_outcome(latest[qid]) not in TERMINAL_OUTCOMES]


def merge_stats(total: dict, part: dict) -> dict:
    """Sum two CallStats.as_dict() snapshots (numbers add, dicts add per key, lists concatenate)."""
    out = dict(total)
    for key, value in part.items():
        if isinstance(value, dict):
            merged = dict(out.get(key) or {})
            for k, v in value.items():
                merged[k] = merged.get(k, 0) + v
            out[key] = merged
        elif isinstance(value, list):
            out[key] = list(out.get(key) or []) + value
        elif isinstance(value, (int, float)):
            out[key] = (out.get(key) or 0) + value
        else:
            out[key] = value
    return out


def run_config(system: System, args) -> dict:
    return {"provider": system.provider, "retrieval": system.retrieval, "mode": system.mode,
            "corpus_system": system.corpus,
            "embedding_model": system.embedding_model, "embedding_dim": system.embedding_dim,
            "chat_model": system.chat_model, "top_k": system.top_k,
            "chunk_size": baseline_a.CHUNK_SIZE, "chunk_overlap": baseline_a.CHUNK_OVERLAP,
            "fresh_query_embeddings": args.fresh_query_embeddings,
            "system_prompt_sha256": hashlib.sha256(system.build_system_prompt("{context}").encode()).hexdigest(),
            "retrieval_sql_sha256": hashlib.sha256(system.retrieval_sql.encode()).hexdigest(),
            "deviations": list(system.deviations),
            "nul_characters_removed_by_paper": {
                p.id: n for p in load_corpus()
                if (n := sum(preprocess.nul_count(t) for t in load_pages(p)))}}


def resume_mismatches(meta: dict, current: dict) -> list[str]:
    """Differences that make a resume inconsistent with the saved run. Config keys
    absent from an older run.json are not mismatches (they are recorded on resume)."""
    problems = []
    for key in ("system", "questions_file_sha256", "included_drafts"):
        if meta.get(key) != current.get(key):
            problems.append(f"{key}: saved {meta.get(key)!r} != current {current.get(key)!r}")
    for key, value in current["config"].items():
        if key in meta["config"] and meta["config"][key] != value:
            problems.append(f"config.{key}: saved {meta['config'][key]!r} != current {value!r}")
    if meta.get("db", {}).get("chunks") != current["db"].get("chunks"):
        problems.append(f"db.chunks: saved {meta.get('db', {}).get('chunks')} != current {current['db'].get('chunks')}")
    return problems


async def _answer_with_graph(system: System, q: dict, retrieve_for, filenames: dict, stats, t_start: float) -> dict:
    """One question through agentic.py; returns trace fields in the harness's usual shape. ``retrieved`` is the
    final 5-chunk context given to synthesis, so retrieval metrics compare equal context budgets."""
    import agentic
    import ollama_backend

    tokens_before = dict(stats.tokens)
    embed = {"seconds": 0.0, "cached": True}

    def retrieve(query: str) -> list[dict]:
        results, embed_s, cache_hit = retrieve_for(query)
        embed["seconds"] += embed_s
        embed["cached"] = embed["cached"] and cache_hit
        return results

    def llm_json(kind, messages, schema):
        return ollama_backend.chat_json(model=system.chat_model, messages=messages, schema=schema, stats=stats,
                                        kind=kind, max_retries=env_int("EVAL_MAX_RETRIES", 2))

    async def generate(system_prompt: str, question: str) -> dict:
        return await ollama_backend.stream_generate(model=system.chat_model, system_instruction=system_prompt,
                                                    question=question, stats=stats,
                                                    max_retries=env_int("EVAL_MAX_RETRIES", 2))

    deps = agentic.GraphDeps(retrieve=retrieve, llm_json=llm_json, generate=generate,
                             source_label=lambda c: filenames.get(c.get("document_id"), "unknown"))
    state = await agentic.run_graph(deps, q["question"])
    gen = state["generation"]
    ttft = gen["started_after_s"] + gen["ttft_s"] if gen.get("ttft_s") is not None else None
    usage = {k: v - tokens_before.get(k, 0) for k, v in stats.tokens.items() if v - tokens_before.get(k, 0)}
    return {
        "retrieved": state["context"],
        "answer": gen.get("text", ""),
        "timing": {"embed_query_s": embed["seconds"], "query_embedding_cached": embed["cached"],
                   "ttft_s": ttft, "generation_s": gen.get("total_s"), "end_to_end_s": time.perf_counter() - t_start,
                   "pre_synthesis_s": gen["started_after_s"], "server_s": gen.get("server_timings_s")},
        "usage": usage, "generation_attempts": gen.get("attempts"), "error": gen.get("error"),
        "graph": {"queries": state["queries"], "rounds": state["rounds"], "assessments": state["assessments"],
                  "llm_calls": state["llm_calls"], "parse_errors": state["parse_errors"],
                  "citations": state["citations"], "events": state["events"]},
    }


async def _answer_questions(system: System, questions: list[dict], conn, client, run_dir, stats, args,
                            session: int, prior_records: list[dict]):
    from google.genai import types

    import db
    from gemini import DailyQuotaExhausted, Pacer, RetryBudgetExhausted, stream_generate

    import ollama_backend

    if system.provider == "gemini":
        from gemini import Embedder

        embedder = Embedder(client, model=system.embedding_model, dim=system.embedding_dim, stats=stats,
                            conn=None if args.fresh_query_embeddings else conn,
                            items_per_min=env_int("EVAL_EMBED_ITEMS_PER_MIN", 90),
                            max_retries=env_int("EVAL_MAX_RETRIES", 4))
    else:
        embedder = make_embedder(system, stats=stats, conn=None if args.fresh_query_embeddings else conn)
    gen_pacer = Pacer(env_float("EVAL_GENERATE_RPM", 8) if system.provider == "gemini" else 0, stats)
    max_questions = getattr(args, "max_questions", None)
    if system.retrieval == "hybrid_bm25":
        import hybrid

        hybrid.assert_lexical_index_fresh(conn)
    if system.mode == "graph" and system.provider != "ollama":
        raise RuntimeError("graph mode is implemented for the local Ollama systems only")
    with conn.cursor() as cur:
        cur.execute("SELECT id, filename FROM documents")
        filenames = dict(cur.fetchall())

    def retrieve_for(query: str) -> tuple[list[dict], float, bool]:
        vec, embed_s, cache_hit = embedder.embed_query(query)
        if system.retrieval == "hybrid_bm25":
            import hybrid

            return hybrid.hybrid_retrieve(conn, user_id=user_id, query_text=query, query_embedding=vec,
                                          top_k=system.top_k, lexical="bm25"), embed_s, cache_hit
        return db.retrieve(conn, system.retrieval_sql, user_id, vec, system.top_k), embed_s, cache_hit
    user_id = db.ensure_eval_user(conn)
    traces_path = run_dir / "traces.jsonl"
    prior_attempts: dict[str, int] = {}
    for record in prior_records:
        prior_attempts[record["question_id"]] = prior_attempts.get(record["question_id"], 0) + 1
    already_answered = {qid for qid, t in latest_traces(prior_records).items() if trace_outcome(t) == ANSWERED}
    attempted: list[str] = []
    with traces_path.open("a", encoding="utf-8") as fh:  # append-only: earlier records are never rewritten
        for n, q in enumerate(questions, start=1):
            if max_questions is not None and len(attempted) >= max_questions:
                log.info("session limit of %d question(s) reached — pausing (resume to continue)", max_questions)
                return {"status": "paused", "stop_reason": None, "attempted": attempted}
            if q["id"] in already_answered:  # duplicate guard (callers already skip these)
                raise RuntimeError(f"{q['id']} already has an answered record — refusing to write a duplicate")
            trace = {"question_id": q["id"], "type": q["type"], "question": q["question"], "error": None,
                     "session": session, "attempt_seq": prior_attempts.get(q["id"], 0) + 1,
                     "started_at": _utc_now()}
            errors_before = len(stats.errors)
            t_start = time.perf_counter()
            stop = None
            try:
                if system.mode == "graph":
                    trace.update(await _answer_with_graph(system, q, retrieve_for, filenames, stats, t_start))
                    gen = {"error": trace["error"]}
                    if gen["error"] and gen["error"].get("retry_budget_exhausted"):
                        stop = ("stopped_retry_budget_exhausted", {"stage": "generate", **gen["error"]})
                else:
                    t0 = time.perf_counter()
                    retrieved, embed_s, cache_hit = retrieve_for(q["question"])
                    db_s = time.perf_counter() - t0 - embed_s
                    prompt = system.build_system_prompt(system.build_context([c["text"] for c in retrieved]))
                    if system.provider == "ollama":
                        gen = await ollama_backend.stream_generate(
                            model=system.chat_model, system_instruction=prompt, question=q["question"], stats=stats,
                            max_retries=env_int("EVAL_MAX_RETRIES", 2))
                    else:
                        contents = [types.Content(role="user", parts=[types.Part(text=q["question"])])]
                        gen = await stream_generate(client, model=system.chat_model, contents=contents,
                                                    system_instruction=prompt, stats=stats, pacer=gen_pacer,
                                                    max_retries=env_int("EVAL_MAX_RETRIES", 4))
                    trace.update({
                        "retrieved": retrieved,
                        "answer": gen["text"],
                        "timing": {"embed_query_s": None if cache_hit else embed_s, "query_embedding_cached": cache_hit,
                                   "retrieval_db_s": db_s, "ttft_s": gen["ttft_s"], "generation_s": gen["total_s"],
                                   "end_to_end_s": time.perf_counter() - t_start},
                        "usage": gen["usage"], "generation_attempts": gen["attempts"], "error": gen["error"],
                    })
                    if gen.get("server_timings_s"):
                        trace["timing"]["server_s"] = gen["server_timings_s"]
                    if gen["error"] and gen["error"].get("daily_quota"):
                        stop = ("stopped_daily_quota", {"stage": "generate", **gen["error"]})
                    elif gen["error"] and gen["error"].get("retry_budget_exhausted"):
                        stop = ("stopped_retry_budget_exhausted", {"stage": "generate", **gen["error"]})
            except ollama_backend.OllamaError as exc:  # local server unavailable after bounded retries
                trace["error"] = {"stage": exc.kind, **exc.details}
                stop = ("stopped_unavailable", {"stage": exc.kind, **exc.details})
            except RetryBudgetExhausted as exc:
                trace["error"] = {"category": exc.category, "type": type(exc).__name__, "stage": exc.kind,
                                  **{k: v for k, v in exc.details.items() if k != "type"}}
                status = "stopped_daily_quota" if isinstance(exc, DailyQuotaExhausted) else "stopped_retry_budget_exhausted"
                stop = (status, {"stage": exc.kind, "category": exc.category, **exc.details,
                                 "quota": exc.quota, "retry_after_s": exc.retry_after_s})
            except Exception as exc:  # noqa: BLE001 — recorded, run continues
                trace["error"] = {"category": "harness_or_embedding", "type": type(exc).__name__}
                log.warning("%s failed: %s", q["id"], type(exc).__name__)
            trace["finished_at"] = _utc_now()
            trace["attempt_errors"] = stats.errors[errors_before:]  # per-question retries/errors (sanitized)
            trace["outcome"] = trace_outcome(trace)
            fh.write(json.dumps(trace) + "\n")
            fh.flush()
            attempted.append(q["id"])
            log.info("[%d/%d] %s %s", n, len(questions), q["id"], trace["outcome"])
            if stop:
                log.error("STOPPING session after %s: %s (%s)", q["id"], stop[0], stop[1])
                return {"status": stop[0], "stop_reason": stop[1], "attempted": attempted}
    return {"status": "session_finished", "stop_reason": None, "attempted": attempted}


def _write_json_atomic(path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def cmd_run(args) -> int:
    import db
    from gemini import CallStats, make_client

    system = SYSTEMS[args.system]
    resuming = bool(args.resume)
    if resuming and (args.questions or args.limit or args.run_id):
        log.error("--resume continues the saved question set; do not combine it with --questions/--limit/--run-id")
        return 1
    if resuming:
        run_dir = RUNS_DIR / args.resume
        if not (run_dir / "run.json").exists():
            log.error("no saved run at %s", run_dir)
            return 1
        meta = json.loads((run_dir / "run.json").read_text())
        system = SYSTEMS[meta["system"]]
        all_questions = {q["id"]: q for q in load_questions()}
        questions = [all_questions[qid] for qid in meta["question_ids"]]
    else:
        questions = [q for q in load_questions() if args.include_drafts or q["status"] == "approved"]
        if args.questions:
            wanted = set(args.questions)
            questions = [q for q in questions if q["id"] in wanted]
        if args.limit:
            questions = questions[: args.limit]
    if not questions:
        log.error("no questions selected — questions must be status 'approved' (or pass --include-drafts)")
        return 1
    conn = db.connect()
    stats_db = db.corpus_stats(conn)
    if not stats_db["chunks"]:
        log.error("no chunks in the eval DB — run `ingest --system %s` first", system.name)
        return 1
    with conn.cursor() as cur:
        cur.execute("SELECT paper_id FROM eval_documents WHERE system = %s", (system.corpus,))
        ingested = {r[0] for r in cur.fetchall()}
    missing = [p.id for p in load_corpus() if p.id not in ingested]
    if missing:
        log.error("corpus incompletely ingested for System %s (missing: %s) — refusing to benchmark a partial "
                  "corpus; finish `ingest` first", system.name, ", ".join(missing))
        return 1
    assert_single_embedding_space(conn, system)

    current = {"system": system.name, "questions_file_sha256": _sha256(QUESTIONS_FILE),
               "included_drafts": meta["included_drafts"] if resuming else args.include_drafts,
               "config": run_config(system, args), "db": stats_db}
    if resuming:
        problems = resume_mismatches(meta, current)
        if problems:
            log.error("refusing to resume %s — configuration changed:\n  %s", args.resume, "\n  ".join(problems))
            return 1
        for key, value in current["config"].items():  # older run.json: record keys it did not have yet
            meta["config"].setdefault(key, value)
        if "sessions" not in meta:  # older single-session run.json -> session 1
            meta["sessions"] = [{"session": 1, "started_at": meta.get("started_at"), "status": meta.get("status"),
                                 "stop_reason": meta.get("stop_reason"), "elapsed_s": meta.get("elapsed_s"),
                                 "attempted": meta.get("completed", []), "stats": meta.get("stats", {})}]
        prior_records = read_traces(run_dir / "traces.jsonl")
        pending = pending_question_ids(meta["question_ids"], prior_records)
        if not pending:
            log.info("run %s already has a terminal outcome for every question — nothing to do", args.resume)
            return 0
        questions = [q for q in questions if q["id"] in set(pending)]
        run_id = meta["run_id"]
        session = len(meta["sessions"]) + 1
    else:
        run_id = args.run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{system.name}"
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        prior_records, session = [], 1
        meta = {"run_id": run_id, "system": system.name, "description": system.description,
                "config": current["config"], "questions_file_sha256": current["questions_file_sha256"],
                "question_ids": [q["id"] for q in questions], "included_drafts": args.include_drafts,
                "db": stats_db, "environment": environment_metadata(), "started_at": _utc_now(), "sessions": []}

    lock = run_dir / ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # one writer per run: no duplicate records
        os.close(fd)
    except FileExistsError:
        log.error("run %s is locked by another process (%s) — refusing to run concurrently", run_id, lock)
        return 1

    if system.provider == "ollama":
        import ollama_backend

        client = None
        if not resuming:
            meta["models"] = ollama_backend.model_metadata([system.chat_model, system.embedding_model])
    else:
        client = make_client()
        if not resuming:
            meta["models"] = _model_metadata(client, [system.chat_model, system.embedding_model])
    stats = CallStats()
    started = time.perf_counter()
    session_record = {"session": session, "started_at": _utc_now(), "status": "failed", "stop_reason": None,
                      "attempted": []}
    log.info("run %s session %d: %d question(s) to answer%s", run_id, session, len(questions),
             f" (resuming; {len(meta['question_ids']) - len(questions)} already terminal)" if resuming else "")
    try:
        outcome = asyncio.run(_answer_questions(system, questions, conn, client, run_dir, stats, args,
                                                session, prior_records))
        session_record.update(outcome)
    except Exception as exc:  # noqa: BLE001 — persisted below
        session_record["stop_reason"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
        log.error("run failed: %s", type(exc).__name__)
    finally:
        session_record["finished_at"] = _utc_now()
        session_record["elapsed_s"] = round(time.perf_counter() - started, 2)
        session_record["stats"] = stats.as_dict()
        meta["sessions"].append(session_record)
        records = read_traces(run_dir / "traces.jsonl")
        latest = latest_traces(records)
        pending = pending_question_ids(meta["question_ids"], records)
        meta["completed"] = [qid for qid in meta["question_ids"] if qid not in pending]  # terminal outcomes
        meta["remaining"] = pending
        meta["outcomes"] = {qid: trace_outcome(latest[qid]) for qid in meta["question_ids"] if qid in latest}
        if not pending:
            meta["status"] = "completed"
        elif session_record["status"] == "paused":
            meta["status"] = "paused"  # --max-questions reached; resume to continue
        elif session_record["status"] == "session_finished":
            meta["status"] = "pending_retry"  # e.g. incomplete answers to regenerate on the next resume
        else:
            meta["status"] = session_record["status"]
        meta["stop_reason"] = session_record["stop_reason"]
        meta["elapsed_s"] = round(sum(s.get("elapsed_s") or 0 for s in meta["sessions"]), 2)
        total: dict = {}
        for s in meta["sessions"]:
            total = merge_stats(total, s.get("stats") or {})
        meta["stats"] = total
        _write_json_atomic(run_dir / "run.json", meta)
        lock.unlink(missing_ok=True)
    log.info("run %s %s after session %d (%d/%d questions terminal, %d pending)%s", run_id, meta["status"], session,
             len(meta["completed"]), len(meta["question_ids"]), len(meta["remaining"]),
             f" — now: run_eval.py report {run_id}" if meta["status"] == "completed"
             else f" — resume with: run_eval.py run --resume {run_id}")
    return 0 if meta["status"] == "completed" else 2


# --- retrieval-only comparison (Phase 2) ---------------------------------------------------
RETRIEVAL_METHODS = ("vector", "hybrid", "hybrid_bm25")  # hybrid = ts_rank_cd lexical list (first design)
SPLIT_FILE = QUESTIONS_FILE.parent / "splits" / "phase2_split.json"


def split_question_ids(questions: list[dict]) -> dict[str, list[str]]:
    """Pre-registered paper-group split (evaluation/splits/phase2_split.json)."""
    split = json.loads(SPLIT_FILE.read_text())
    out = {}
    for subset in ("dev", "heldout"):
        papers = {p for g in split[f"{subset}_groups"] for p in split["groups"][g]}
        out[subset] = [q["id"] for q in questions if q.get("papers") and set(q["papers"]) <= papers]
    return out


def compare_retrieval(conn, system: System, questions: list[dict], embedder, user_id: int, ks=(5, 10)) -> dict:
    """Score vector-only (the system's own SQL) and hybrid retrieval on the same chunks and the same
    query embeddings. No generation — only the retrieval step differs between the two methods."""
    import db
    import hybrid

    top = max(ks)
    per_question, latency = [], {m: [] for m in RETRIEVAL_METHODS}
    for q in questions:
        vec, _, _ = embedder.embed_query(q["question"])
        row = {"question_id": q["id"], "type": q["type"]}
        t0 = time.perf_counter()
        results = {"vector": db.retrieve(conn, system.retrieval_sql, user_id, vec, top)}
        latency["vector"].append(time.perf_counter() - t0)
        for method, lexical in (("hybrid", "tsrank"), ("hybrid_bm25", "bm25")):
            t0 = time.perf_counter()
            results[method] = hybrid.hybrid_retrieve(conn, user_id=user_id, query_text=q["question"],
                                                     query_embedding=vec, top_k=top, lexical=lexical)
            latency[method].append(time.perf_counter() - t0)
        for method, retrieved in results.items():
            row[method] = M.score_retrieval(q, retrieved, ks=ks)
            row[method]["chunk_ids"] = [r["chunk_id"] for r in retrieved]
            if method != "vector":
                row[method]["from_fts_only"] = sum(1 for r in retrieved[:5] if r["vector_rank"] is None)
        per_question.append(row)
    metric_keys = [k for k in per_question[0]["vector"] if k not in ("chunk_ids", "from_fts_only")] \
        if per_question else []

    def summarise(rows):
        out = {}
        for method in RETRIEVAL_METHODS:
            out[method] = {key: dict(zip(("mean", "n"), M.mean_defined([r[method].get(key) for r in rows])))
                           for key in metric_keys}
            if method != "vector":  # paired comparison with the vector control at recall@5
                pairs = [(r[method].get("evidence_recall@5"), r["vector"].get("evidence_recall@5")) for r in rows]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                out[method]["vs_vector_recall@5"] = {"better": sum(a > b for a, b in pairs),
                                                     "worse": sum(a < b for a, b in pairs),
                                                     "tied": sum(a == b for a, b in pairs)}
        return out

    summary = summarise(per_question)
    for method in RETRIEVAL_METHODS:
        summary[method]["latency_s"] = {"p50": M.percentile(latency[method], 50),
                                        "p95": M.percentile(latency[method], 95)}
    subsets = {}
    if SPLIT_FILE.exists():
        for subset, ids in split_question_ids(questions).items():
            rows = [r for r in per_question if r["question_id"] in set(ids)]
            subsets[subset] = {"questions": len(rows), **summarise(rows)}
    by_type = {}
    for qtype in sorted({r["type"] for r in per_question}):
        rows = [r for r in per_question if r["type"] == qtype]
        by_type[qtype] = {m: {key: M.mean_defined([r[m].get(key) for r in rows])[0]
                              for key in ("evidence_recall@5", "evidence_mrr")} for m in RETRIEVAL_METHODS}
        by_type[qtype]["n"] = len(rows)
    return {"summary": summary, "subsets": subsets, "by_type": by_type, "per_question": per_question}


def cmd_retrieval_eval(args) -> int:
    import db
    import hybrid
    from gemini import CallStats

    system = SYSTEMS[args.system]
    conn = db.connect()
    assert_single_embedding_space(conn, system)
    with conn.cursor() as cur:
        cur.execute("SELECT paper_id FROM eval_documents WHERE system = %s", (system.corpus,))
        ingested = {r[0] for r in cur.fetchall()}
    missing = [p.id for p in load_corpus() if p.id not in ingested]
    if missing:
        log.error("corpus incompletely ingested for System %s (missing: %s)", system.name, ", ".join(missing))
        return 1
    questions = [q for q in load_questions() if args.include_drafts or q["status"] == "approved"]
    hybrid.assert_lexical_index_fresh(conn)
    stats = CallStats()
    embedder = make_embedder(system, stats=stats, conn=conn)
    result = compare_retrieval(conn, system, questions, embedder, db.ensure_eval_user(conn))
    result["meta"] = {"system": system.name, "provider": system.provider, "embedding_model": system.embedding_model,
                      "questions": len(questions), "questions_file_sha256": _sha256(QUESTIONS_FILE),
                      "hybrid": {"candidates_per_list": hybrid.CANDIDATES, "rrf_k": hybrid.RRF_K,
                                 "fts_config": "english, OR of question lexemes, ts_rank_cd, GIN expression index"},
                      "hybrid_bm25": {"candidates_per_list": hybrid.CANDIDATES, "rrf_k": hybrid.RRF_K,
                                      "k1": hybrid.BM25_K1, "b": hybrid.BM25_B,
                                      "match": "same FTS @@ match (GIN index); BM25 ranking from eval_bm25_* stats"},
                      "split_file_sha256": _sha256(SPLIT_FILE) if SPLIT_FILE.exists() else None,
                      "vector": "system retrieval SQL (pgvector <=>)", "db": db.corpus_stats(conn),
                      "embedding_usage": {k: stats.as_dict()[k] for k in ("requests", "attempts", "retries")},
                      "created_at": _utc_now(), "environment": environment_metadata()}
    name = f"retrieval_{system.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2, default=str))
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARIES_DIR / f"{name}.json").write_text(json.dumps(
        {k: result[k] for k in ("meta", "summary", "subsets", "by_type")}, indent=2, default=str))
    keys = ("evidence_recall@5", "evidence_recall@10", "page_recall@5", "evidence_precision@5", "evidence_mrr",
            "paper_recall@5", "facet_coverage@5")
    print(f"System {system.name} ({system.embedding_model}), {len(questions)} questions — wrote {name}.json")
    for label, block in [("all", result["summary"])] + list(result["subsets"].items()):
        print(f"\n[{label}]" + (f" {block['questions']} questions" if "questions" in block else ""))
        print(f"{'metric':22}" + "".join(f"{m:>16}" for m in RETRIEVAL_METHODS))
        for key in keys:
            print(f"{key:22}" + "".join(f"{_fmt(block[m][key]):>16}" for m in RETRIEVAL_METHODS))
        for m in RETRIEVAL_METHODS[1:]:
            print(f"  {m} vs vector (recall@5 per question): {block[m]['vs_vector_recall@5']}")
    for m in RETRIEVAL_METHODS:
        lat = result["summary"][m]["latency_s"]
        print(f"latency {m:12} p50 {lat['p50']:.3f}s  p95 {lat['p95']:.3f}s")
    return 0


def cmd_build_lexical_index(args) -> int:
    import db
    import hybrid

    conn = db.connect()
    info = hybrid.build_lexical_index(conn)
    log.info("BM25 statistics rebuilt: %s", {k: v for k, v in info.items() if k != "fingerprint"})
    return 0


# --- report ---------------------------------------------------------------------------
def _load_human_scores(run_dir) -> dict[str, dict]:
    path = run_dir / "human_scores.csv"
    if not path.exists():
        return {}
    scores = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("correctness", "").strip() != "":
                scores[row["question_id"]] = row
    return scores


def cmd_report(args) -> int:
    run_dir = RUNS_DIR / args.run_id
    meta = json.loads((run_dir / "run.json").read_text())
    questions = {q["id"]: q for q in load_questions()}
    records = read_traces(run_dir / "traces.jsonl")
    latest = latest_traces(records)  # a question is scored on its latest (final) attempt only
    traces = [latest[qid] for qid in meta["question_ids"] if qid in latest]
    attempts_by_q: dict[str, int] = {}
    for record in records:
        attempts_by_q[record["question_id"]] = attempts_by_q.get(record["question_id"], 0) + 1
    human = _load_human_scores(run_dir)
    k = meta["config"]["top_k"]

    per_question = []
    for t in traces:
        q = questions[t["question_id"]]
        outcome = trace_outcome(t)
        row = {"question_id": t["question_id"], "type": t["type"], "outcome": outcome,
               "error": t["error"] if outcome != ANSWERED else None,
               "trace_records": attempts_by_q[t["question_id"]],
               "generation_attempts": t.get("generation_attempts")}
        if t.get("retrieved") is not None:
            row.update(M.score_retrieval(q, t["retrieved"], ks=(k,)))
            row.update(M.score_answer(q, t.get("answer", "")))
            if t.get("graph"):
                row.update(M.score_citations(q, t["graph"]["citations"], t["retrieved"]))
                row["llm_calls"] = sum(t["graph"]["llm_calls"].values())
                row["retrieval_rounds"] = t["graph"]["rounds"]
            row["timing"] = t["timing"]
            row["usage"] = t.get("usage", {})
        if t["question_id"] in human:
            h = human[t["question_id"]]
            row["human_correctness"] = float(h["correctness"]) / 2.0  # rubric 0/1/2 -> 0..1
            if h.get("paper_attribution", "").strip():
                row["human_paper_attribution"] = float(h["paper_attribution"])
        per_question.append(row)

    def agg(key, rows=per_question):
        value, n = M.mean_defined([r.get(key) for r in rows])
        return {"mean": value, "n": n}

    retrieval_keys = [f"evidence_recall@{k}", f"page_recall@{k}", f"evidence_precision@{k}", f"page_precision@{k}",
                      f"paper_recall@{k}", f"facet_coverage@{k}", "evidence_mrr", "page_mrr"]
    by_type = {}
    for qtype in sorted({r["type"] for r in per_question}):
        rows = [r for r in per_question if r["type"] == qtype]
        by_type[qtype] = {"n": len(rows), **{key: agg(key, rows) for key in retrieval_keys}}

    def lat(key):
        values = [r["timing"][key] for r in per_question if r.get("timing") and r["timing"].get(key) is not None]
        return {"p50": M.percentile(values, 50), "p95": M.percentile(values, 95), "n": len(values)}

    stats = meta["stats"]
    answered = [r for r in per_question if not r["error"]]
    summary = {
        "run_id": meta["run_id"], "system": meta["system"], "description": meta["description"],
        "provider": meta["config"].get("provider", "gemini"),
        "config": meta["config"], "questions": len(per_question), "answered_without_error": len(answered),
        "retrieval": {key: agg(key) for key in retrieval_keys},
        "retrieval_by_type": by_type,
        "answers": {
            "keyword_recall_heuristic": agg("keyword_recall"),
            "negative_abstention_rate_heuristic": agg("abstention_correct"),
            "false_abstention_rate_heuristic": agg("false_abstention"),
            "citation_markers_mean": agg("citation_markers"),
            "citation_validity": agg("citation_validity"),
            "citation_support_precision": agg("citation_support_precision"),
            "answers_with_citations": agg("answer_has_citations"),
            "llm_calls_per_question": agg("llm_calls"),
            "retrieval_rounds_mean": agg("retrieval_rounds"),
            "human_correctness": agg("human_correctness"),
            "human_paper_attribution": agg("human_paper_attribution"),
        },
        "latency_s": {key: lat(key) for key in ("embed_query_s", "retrieval_db_s", "ttft_s", "generation_s", "end_to_end_s")},
        "run_status": {"status": meta.get("status", "unknown"), "completed": len(meta.get("completed", [])),
                       "planned": len(meta.get("question_ids", [])), "remaining": meta.get("remaining", []),
                       "stop_reason": meta.get("stop_reason"),
                       "sessions": [{k2: s.get(k2) for k2 in ("session", "started_at", "finished_at", "status")}
                                    for s in meta.get("sessions", [])],
                       "outcomes": {o: sum(1 for r in per_question if r["outcome"] == o)
                                    for o in sorted({r["outcome"] for r in per_question})},
                       "questions_with_multiple_attempts": {r["question_id"]: r["trace_records"]
                                                            for r in per_question if r["trace_records"] > 1}},
        "gemini": {"successful_requests": stats["requests"], "attempts": stats["attempts"],
                   "retries": stats["retries"], "retry_sleep_s": round(stats.get("retry_sleep_s", 0.0), 1),
                   "pacing_sleep_s": round(stats.get("pacing_sleep_s", 0.0), 1),
                   "embed_items_embedded": stats.get("embed_items_embedded"),
                   "tokens": stats["tokens"],
                   "first_attempt_errors": stats["first_attempt_errors"], "final_failures": stats["final_failures"]},
        "errors": {"questions_with_errors": len(per_question) - len(answered),
                   "categories": sorted({(r["error"] or {}).get("category") for r in per_question if r["error"]})},
        "availability": {
            "retrieval_quality": "measured (deterministic, against annotated gold evidence)",
            "evidence_coverage": "measured (facet coverage@k)",
            "citation_support": ("measured: cited sources mapped to chunk ids; support = cited chunk contains a gold "
                                 "evidence quote (strict — a correct paraphrase elsewhere counts as unsupported)"
                                 if meta["config"].get("mode") == "graph" else
                                 "NOT APPLICABLE — this system is not prompted to cite and returns no source ids"),
            "answer_correctness": ("measured from human_scores.csv" if human else
                                   "UNAVAILABLE — pending human review (fill human_scores.csv, re-run report)"),
            "paper_attribution_in_answers": ("measured from human_scores.csv" if any(
                "human_paper_attribution" in r for r in per_question) else "UNAVAILABLE — pending human review"),
            "keyword_recall / abstention": "heuristic signals only — not correctness",
            "tokens": ("measured from Gemini usage_metadata for generation; embedding token counts not reported by the API"
                       if meta["config"].get("provider", "gemini") == "gemini" else
                       "measured from Ollama prompt_eval_count / eval_count (no thinking tokens: think=false)"),
            "latency": "measured; excludes client-side pacing; query embedding excluded when served from cache",
        },
        "environment": meta["environment"], "models": meta.get("models"), "elapsed_s": meta.get("elapsed_s"),
        "deviations": meta["config"].get("deviations", []),
        "nul_characters_removed_by_paper": meta["config"].get("nul_characters_removed_by_paper", {}),
    }
    (run_dir / "metrics.json").write_text(json.dumps({"summary": summary, "per_question": per_question}, indent=2))

    review = run_dir / "review.csv"
    if not review.exists():
        with review.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["question_id", "type", "question", "gold_answer", "scoring_rule", "answer",
                             "correctness", "paper_attribution", "notes"])
            for t in traces:
                q = questions[t["question_id"]]
                writer.writerow([t["question_id"], t["type"], q["question"], q["gold_answer"],
                                 q.get("scoring_rule", ""), t.get("answer", ""), "", "", ""])

    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARIES_DIR / f"{meta['run_id']}.json").write_text(json.dumps(summary, indent=2))
    md = render_markdown(summary)
    (run_dir / "report.md").write_text(md)
    (SUMMARIES_DIR / f"{meta['run_id']}.md").write_text(md)
    print(md)
    return 0


def _fmt(cell) -> str:
    if isinstance(cell, dict):
        if cell.get("mean") is None:
            return f"n/a (n={cell.get('n', 0)})"
        return f"{cell['mean']:.3f} (n={cell['n']})"
    return "n/a" if cell is None else str(cell)


def render_markdown(s: dict) -> str:
    rs = s.get("run_status", {})
    lines = [f"# Evaluation run `{s['run_id']}` — System {s['system']}", "", s["description"], "",
             f"Run status: **{rs.get('status', 'unknown')}** ({rs.get('completed', '?')}/{rs.get('planned', '?')} "
             f"questions)" + (f" — stopped: {rs['stop_reason']}" if rs.get("stop_reason") else ""), "",
             f"Questions: {s['questions']} (answered without error: {s['answered_without_error']})", "",
             f"Sessions: {len(rs.get('sessions', []))} · outcomes: {rs.get('outcomes', {})} · "
             f"questions re-attempted across sessions: {rs.get('questions_with_multiple_attempts') or 'none'}", "",
             "## Retrieval (top-k = %d)" % s["config"]["top_k"], "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in s["retrieval"].items()]
    lines += ["", "## Answers", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in s["answers"].items()]
    lines += ["", "## Latency (seconds)", "", "| Stage | p50 | p95 | n |", "|---|---|---|---|"]
    for k, v in s["latency_s"].items():
        p50 = "n/a" if v["p50"] is None else f"{v['p50']:.3f}"
        p95 = "n/a" if v["p95"] is None else f"{v['p95']:.3f}"
        lines.append(f"| {k} | {p50} | {p95} | {v['n']} |")
    g = s["gemini"]
    lines += ["", f"## Model usage ({s.get('provider', 'gemini')})", "", f"- Successful requests: {g['successful_requests']}",
              f"- Attempts: {g['attempts']} · retries: {g['retries']} ({g.get('retry_sleep_s', 0)}s waiting) "
              f"· client pacing {g.get('pacing_sleep_s', 0)}s",
              f"- Embedding items embedded this run: {g.get('embed_items_embedded')}",
              f"- Tokens: {g['tokens'] or 'n/a'}",
              f"- First-attempt errors: {g['first_attempt_errors'] or 'none'}",
              f"- Final failures: {g['final_failures'] or 'none'}", "", "## Deviations from production", ""]
    lines += [f"- {d}" for d in s.get("deviations", [])]
    affected = s.get("nul_characters_removed_by_paper") or {}
    if affected:
        lines.append("- NUL characters removed (production would most likely have FAILED to ingest these papers, "
                     "because PostgreSQL text cannot store NUL): "
                     + ", ".join(f"{p} ({n})" for p, n in affected.items()))
    lines += ["", "## Metric availability", ""]
    lines += [f"- **{k}**: {v}" for k, v in s["availability"].items()]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db").set_defaults(func=cmd_init_db)
    sub.add_parser("extract").set_defaults(func=cmd_extract)
    sub.add_parser("validate-annotations").set_defaults(func=cmd_validate)
    p = sub.add_parser("estimate")
    p.add_argument("--system", default="A", choices=sorted(SYSTEMS))
    p.add_argument("--include-drafts", action="store_true")
    p.set_defaults(func=cmd_estimate)
    p = sub.add_parser("check-embedding-equivalence")
    p.add_argument("--system", default="A", choices=sorted(SYSTEMS))
    p.add_argument("--samples", type=int, default=3)
    p.set_defaults(func=cmd_check_equivalence)
    p = sub.add_parser("ingest")
    p.add_argument("--system", default="A", choices=sorted(SYSTEMS))
    p.add_argument("--embed-mode", default="batch", choices=["batch", "single"])
    p.add_argument("--papers", nargs="*")
    p.add_argument("--reingest", action="store_true")
    p.set_defaults(func=cmd_ingest)
    p = sub.add_parser("run")
    p.add_argument("--system", default="A", choices=sorted(SYSTEMS))
    p.add_argument("--questions", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--include-drafts", action="store_true")
    p.add_argument("--fresh-query-embeddings", action="store_true",
                   help="bypass the embedding cache for queries (measures embed latency, costs 1 request/question)")
    p.add_argument("--run-id")
    p.add_argument("--max-questions", type=int, metavar="N",
                   help="answer at most N questions in this session, then pause (resume continues the same run)")
    p.add_argument("--resume", metavar="RUN_ID",
                   help="continue a saved run: same questions and config; only questions without a terminal "
                        "outcome (unanswered, failed, incomplete) are attempted; earlier records are kept")
    p.set_defaults(func=cmd_run)
    sub.add_parser("build-lexical-index", help="(re)build the FTS GIN index + BM25 statistics in the eval DB"
                   ).set_defaults(func=cmd_build_lexical_index)
    p = sub.add_parser("retrieval-eval", help="vector-only vs hybrid retrieval on the same chunks (no generation)")
    p.add_argument("--system", default="A", choices=sorted(SYSTEMS))
    p.add_argument("--include-drafts", action="store_true")
    p.set_defaults(func=cmd_retrieval_eval)
    p = sub.add_parser("report")
    p.add_argument("run_id")
    p.set_defaults(func=cmd_report)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
