# Architecture

Status legend: **Implemented** (in the code today) · **Planned** (in the upgrade plan, not built).

## 1. Production system (System A) — Implemented

```
Browser (Next.js 16, Vercel) ──Supabase Auth (email+password, Google OAuth)──▶ Supabase
   │  Bearer <Supabase access token>
   ▼
FastAPI (Render)  ── verifies token via JWKS/ES256 (api/core/supabase_auth.py), maps sub → users.id
   │ /upload: store file, enqueue Celery task          │ /chat: embed query → pgvector top-5 → Gemini stream
   ▼                                                   ▼
Celery worker (Upstash Redis broker)            PostgreSQL + pgvector (Supabase)
   extract (PyPDF2) → chunk (1000/100)             users, documents, document_chunks(Vector(768)),
   → embed (gemini-embedding-2, 768-d) → insert    chat_sessions, chat_messages
```

**Ingestion flow** (`worker/tasks.py`): `extract_text` concatenates PyPDF2 page text with `"\n"` →
`RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)` → one `embed_content` call
per chunk (`output_dimensionality=768`, no task type) → rows in `document_chunks` → status `completed`.

**Chat flow** (`api/routers/chat.py`): embed the query (same model/dim) → `ORDER BY embedding <=> q
LIMIT 5`, filtered by `documents.user_id` → chunks joined into the `system_instruction` → prior
messages replayed → `client.aio.models.generate_content_stream` (`gemini-3.6-flash`) → chunked
`StreamingResponse`; session id in the `X-Session-ID` header; assistant message saved after the stream.

**Tenant isolation:** application-layer `user_id` filters only — no Postgres row-level security.

## 2. Evaluation harness — Implemented (Phase 1)

```
corpus.yaml ─▶ download_corpus.py (arXiv, sha256 lock) ─▶ data/pdfs/
                         run_eval.py extract ─▶ data/pages/*.json (per-page text)
questions.jsonl (42 approved, gold quotes + pages) ─▶ validate-annotations

run_eval.py ingest ─▶ baseline_a.py (frozen copy of production at 4a89dc1)
                      + preprocess.strip_nul (evaluation-only)
                      + provenance.py (chunk → PDF pages, for scoring only)
                      + gemini.Embedder (batched, item-paced, cached, bounded retries)
                   ─▶ local pgvector (Docker, 127.0.0.1:5434, db.py guard refuses anything else)

run_eval.py run  ─▶ per question: cached/embedded query → same SQL as production → same prompt
                   → streamed generation → append-only traces.jsonl + run.json (sessions)
run_eval.py report ─▶ metrics.py → metrics.json, report.md, review.csv (human rubric)
```

Isolation: the harness never reads `DATABASE_URL`, never touches Supabase, and is excluded from
the production image (`.dockerignore`). See [evaluation.md](evaluation.md).

## 3. Local-inference option — Implemented on branch `feature/local-inference-qwen3-14b`, not yet exercised end-to-end

```
Mac: FastAPI / worker / local_inference/check.py
   │  core/inference.py: get_provider()  ── INFERENCE_PROVIDER=gemini (default) → google-genai (unchanged)
   │                                      └─ INFERENCE_PROVIDER=ollama → httpx (trust_env=False)
   ▼  http://127.0.0.1:11435  (SSH -L, control socket ~/.ssh/documind-ollama-tunnel.sock)
lab server `<lab-host>`: system Ollama 0.20.3 bound to 127.0.0.1:11434 (not exposed)
   ├─ /api/embed  nomic-embed-text (137M, F16, 768-d)   — "search_document:" / "search_query:" prefixes
   └─ /api/chat   qwen3:14b (14.8B, Q4_K_M, installed)    — NDJSON stream, think=false, num_ctx 8192
   ▼
local pgvector only (DATABASE_URL must be loopback) — never the shared Gemini-vector database
```
* Same chunking, retrieval SQL and prompt as System A; only the two model calls change.
* No fallback between providers: an unreachable Ollama raises `InferenceUnavailable`.
* Worker: `worker/inference.py` (embedding-only twin; the worker is a separate image).
* Verified so far: tunnel, server version, metadata of both models (`nomic-embed-text` 768-d / 2,048 ctx;
  `qwen3:14b` 14.77B Q4_K_M, 40,960 ctx).
  Not verified: any embedding/generation call, latency, VRAM use — see local_inference/README.md.

**Two inference environments (same code, different env file):** lab server `<lab-host>` — `qwen3:14b` via SSH
tunnel (`local_inference/env.example`); Mac — `qwen3:8b` on the Mac's own Ollama at 127.0.0.1:11434
(`local_inference/env.mac.example`, branch `feature/local-inference-qwen3-8b-mac`). Both use
`nomic-embed-text` and an isolated local pgvector on port 5436. Mac path verified end-to-end on synthetic data
(embed → pgvector retrieval → qwen3:8b streamed answer); lab path still awaiting GPU headroom.

## 4. Evaluation-only Phase 2–3 components — Implemented in `evaluation/`, not in the application

```
question ─▶ agentic.py (LangGraph, CL/CLV)                         one-shot systems (A, L, BL)
            plan (JSON) ─▶ retrieve ◀─┐                            retrieve ─▶ prompt ─▶ answer
                            │         │ follow-up (≤ 2 rounds)
                            ▼         │
                          assess (JSON)
                            ▼
                     synthesize (5 sources, [S#]) ─▶ verify (deterministic)
retrieve = vector (System A SQL) | BM25 hybrid (hybrid.py: pgvector + FTS/GIN + BM25 + RRF)
```
* Systems: L (vector, one-shot) · BL (BM25 hybrid, one-shot) · CL (graph, BM25 hybrid) · CLV (graph, vector) —
  all qwen3:8b + nomic-embed-text on System L's chunks; only retrieval and/or orchestration differ.

## 5. Planned upgrade (application) — Planned, not implemented

* **System B (Advanced RAG):** page/section-aware ingestion (page numbers stored), hybrid
  retrieval = pgvector + PostgreSQL full-text search fused with Reciprocal Rank Fusion, citations.
* **System C (Agentic):** bounded LangGraph workflow (retrieve → assess evidence → optionally
  re-query → answer with verified citations), multi-paper comparison.
* Schema migrations and a production re-index are prerequisites (Phase 2–3).

## 6. Retrieval modes in the application (implemented on the working branch; not deployed)
`api/core/retrieval.py`: `RETRIEVAL_MODE=vector` (default, original query) or `hybrid_bm25` (vector + PostgreSQL
full-text + BM25 + RRF in one statement, per-user statistics, savepoint + timeout fallback to vector). Requires migration
`scripts/migrations/bm25.py` (nullable `content_tsv`/`token_count` columns, trigger, GIN index). Validated on local
staging only — see docs/proposals/bm25-hybrid-production.md.
