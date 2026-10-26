# Architecture

Status legend: **Implemented** (in the code today) · **Planned** (in the upgrade plan, not built).
Sections 1–2 describe the original hosted baseline (named DocuMind, since retired); section 5 is the
research-paper analyst. The current local setup (Docker Compose + local Supabase + Ollama) is drawn in the
README's Architecture section.

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

## 3. Local inference (Ollama) — Implemented; the default local route. Remote lab-GPU inference unverified

```
Mac: FastAPI / worker / local_inference/check.py
   │  core/inference.py: get_provider()  ── INFERENCE_PROVIDER=gemini (default) → google-genai (unchanged)
   │                                      └─ INFERENCE_PROVIDER=ollama → httpx (trust_env=False)
   ▼  http://127.0.0.1:11435  (SSH -L, control socket ~/.ssh/documind-ollama-tunnel.sock)
lab server `<lab-host>`: system Ollama 0.20.3 bound to 127.0.0.1:11434 (not exposed)
   ├─ /api/embed  nomic-embed-text (137M, F16, 768-d)   — "search_document:" / "search_query:" prefixes
   └─ /api/chat   qwen3:14b (14.8B, Q4_K_M, installed)    — NDJSON stream, think=false, num_ctx 8192
   ▼
local database only (loopback, host.docker.internal or a Compose service) — never a hosted database
```
* Same chunking, retrieval SQL and prompt as System A; only the two model calls change.
* No fallback between providers: an unreachable Ollama raises `InferenceUnavailable`.
* Worker: `worker/inference.py` (embedding-only twin; the worker is a separate image).
* Lab path, verified earlier: tunnel, server version, metadata of both models (`nomic-embed-text` 768-d / 2,048 ctx;
  `qwen3:14b` 14.77B Q4_K_M, 40,960 ctx).
  Not verified: any embedding/generation call on the lab GPU, latency, VRAM use (first blocked by GPU headroom,
  later SSH to the server timed out) — see local_inference/README.md.
* Embedding spaces are recorded per document (`documents.embedding_model`, migration 003) and retrieval only
  searches the active space, so vectors from different models are never compared.

**Two inference environments (same code, different env file):** lab server `<lab-host>` — `qwen3:14b` via SSH
tunnel (`local_inference/env.example`); Mac — `qwen3:8b` on the Mac's own Ollama at 127.0.0.1:11434
(`local_inference/env.mac.example`, branch `feature/local-inference-qwen3-8b-mac`). Both use
`nomic-embed-text` and an isolated local pgvector on port 5436. Mac path verified end-to-end, first on synthetic data and later
through the full application from a clean copy of the repository (README Quick start); the lab path remains
unverified.

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

## 5. Research-paper analyst (application) — Implemented; validated locally (not deployed)

```
upload ─▶ validate (type, size, %PDF) ─▶ Storage ─▶ worker: per-page PyPDF2 ─▶ headings ─▶ 1000/100 split
          within page+section ─▶ batched/paced embeddings ─▶ ONE transaction: replace chunks + metadata

chat ─▶ ownership check (chat, document_ids) ─▶ [rag] embed ─▶ retrieve (user ∩ chat ∩ selected) ─┐
                                        └ [agent] plan ─▶ retrieve ─▶ coverage ─▶ assess ┤ (≤ 2 rounds, budget)
                                                                                          ▼
        numbered sources [S#] (title, page, section) ─▶ guarded prompt ─▶ streamed answer ─▶ citations checked,
        X-Sources header                                                                     stored with message

compare/gaps ─▶ per-paper retrieval (aspect sections, then whole paper) ─▶ JSON per paper ─▶ Pydantic +
                citation validation ─▶ table cells cited or "not reported in the paper"; gaps labelled by cited papers:
                ≥ 2 = cross-paper hypothesis, 1 = single-paper observation (one bounded follow-up to link them)
```

* Schema: migrations `002_provenance`, `003_embedding_space` and `004_chat_documents` (scripts/migrate.py);
  `init_db.sql` matches them for new databases, and the API creates the same schema on first start.
* Chat-scoped papers: `documents.session_id` → `chat_sessions.id` (ON DELETE CASCADE). Every paper endpoint
  and every retrieval query (vector SQL, BM25 hybrid SQL incl. its statistics, agent, compare, gaps) filters on user
  **and** chat; `core/ownership.py` returns 404 for another user's or another chat's ids. The worker receives the
  chat id with each task and never changes it. Legacy papers (NULL chat) are searched nowhere until the user moves
  one into a chat (`POST /document/{id}/assign`, only from NULL; papers are never shared).
* Modules: `worker/ingest.py`, `worker/inference.py`, `api/core/{sources,ownership,agent,observability,schema_check}.py`,
  `api/routers/{chat,analysis}.py`; frontend `components/{Sources,ComparePanel}.tsx`.
* Defaults: `RETRIEVAL_MODE=vector`, `AGENT_ENABLED=false`, reranking not implemented (plan: off).
* Release and rollback: [release.md](release.md).

## 6. Retrieval modes in the application (implemented; not deployed)
`api/core/retrieval.py`: `RETRIEVAL_MODE=vector` (default, original query) or `hybrid_bm25` (vector + PostgreSQL
full-text + BM25 + RRF in one statement, per-user statistics, savepoint + timeout fallback to vector). Requires migration
`scripts/migrations/bm25.py` (nullable `content_tsv`/`token_count` columns, trigger, GIN index). Validated on local
staging only — see docs/proposals/bm25-hybrid-production.md.
