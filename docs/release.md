# Release and rollback — research-paper analyst (plan Phase 5)

> **Historical.** Written for the hosted deployment that has since been retired. ResearchLens is now run
> locally (README "Quick start"); the migration and rollback mechanics below still apply to any existing
> database.

Status: **validated locally only.** Nothing in this document has been run against production. Every step
marked *approval* needs an explicit go-ahead from the project owner before it is run.

## What changes in this release

| Area | Change | Default | Off switch |
| --- | --- | --- | --- |
| Schema | migration `002_provenance` (paper metadata, chunk page/section/position, `chat_messages.sources`, cascade FK, two indexes) | required | `scripts/migrate.py down 002_provenance` |
| Ingestion | page/section-aware chunks, batched + paced embeddings, one-transaction replace, `documents.error` | on | redeploy the previous worker image |
| Chat | numbered sources `[S#]` in the prompt, `X-Sources` header, citations stored, paper selection, safe failure | on | redeploy the previous API image |
| Deep analysis | bounded LangGraph agent (`mode: "agent"`) | **off** (`AGENT_ENABLED=false`) | leave the flag off |
| Analysis | `POST /analysis/compare`, `POST /analysis/gaps` | on | not linked when no papers are selected; previous image removes them |
| Uploads | PDF/TXT only, `%PDF` check, `MAX_UPLOAD_MB` (20) | on | raise the env var |
| Logs | `key=value` lines with request id; `X-Request-ID` header | on | `LOG_LEVEL=WARNING` |
| Retrieval | unchanged: `RETRIEVAL_MODE=vector` | vector | — (BM25 is a separate proposal) |

## Pre-release checks (local)

1. API tests (`api/`, 75) and worker tests (`worker/tests`, 36) pass against a throwaway pgvector database. CI runs both.
2. `cd frontend && npm run lint && npm run build`.
3. Local end-to-end on the Supabase CLI stack with Ollama (no Gemini calls): upload → metadata and sections →
   cited chat → paper-restricted chat → deep analysis → compare → gaps; results in `docs/evaluation.md`.
4. `python scripts/migrate.py status` on a copy of the production schema (restored dump on 127.0.0.1) shows
   `002_provenance` applying cleanly; time each step.

## Production procedure (*approval* at each step)

The migration runner only connects to loopback databases on purpose. For production, the owner either runs
it over a tunnel that exposes the database on 127.0.0.1, or applies the statements in
`scripts/migrations/provenance.py` by hand in the Supabase SQL editor (the `CREATE INDEX CONCURRENTLY` statements
must run outside a transaction, one at a time).

1. **Backup** — Supabase point-in-time recovery or `pg_dump` of `documents`, `document_chunks`, `chat_messages`.
2. **Migrate first** (old code ignores the new nullable columns, so the schema can lead the code):
   `python scripts/migrate.py up --database-url <tunnelled url>`; then `status` must say `complete: True`.
3. **Deploy the worker, then the API** (the API refuses to start if the schema is behind — that is the guard).
4. **Smoke test** with a test account: upload one PDF, wait for `completed`, check title/sections in the
   sidebar, ask a question and confirm cited sources, run a two-paper comparison.
5. **Watch logs** for `event=inference_error`, `event=ingest_failed`, `event=agent_fallback`, 5xx `event=request`.
6. **Re-index (optional, approval)** — existing documents have no page/section provenance. Re-uploading a paper
   re-ingests it; a bulk re-index costs one embedding per chunk (Gemini free tier: ~1,000 texts/day), so plan it
   over several days or on a paid quota. Old chunks stay searchable meanwhile.
7. **Deep analysis** stays off. Turning it on is a separate decision: the evaluation (TD-17) measured ~2–3× tokens
   and latency with no retrieval gain; with Gemini's free tier each agent answer uses 3 extra requests.

The clean reset in the plan (Phase 1: wipe and recreate production data) is **not** part of this procedure and
has not been approved; the migration path above makes it unnecessary.

## Rollback

| Situation | Action | Data impact |
| --- | --- | --- |
| New API misbehaves | redeploy the previous API image | none — old code ignores new columns |
| New worker misbehaves | redeploy the previous worker image | new uploads get legacy chunks (no provenance) |
| Agent misbehaves | `AGENT_ENABLED=false` (default) | none |
| Schema must go | deploy the previous API/worker **first**, then `migrate.py down 002_provenance` | drops metadata, provenance and stored citations; restores the non-cascading FK |

`down` is the last resort: it is destructive for the new columns. Rolling back code alone is always enough to
restore the previous behaviour.
