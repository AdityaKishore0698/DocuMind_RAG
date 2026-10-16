# Proposal — BM25 hybrid retrieval in production (revision 2, 2026-10-16)

**Status:** implemented and validated locally/staging (evidence: §8); **not deployed**. Approved for local/staging implementation only. Production access, migration and deployment need a
separate explicit approval. Retrieval only (vector + PostgreSQL full-text + BM25 + RRF) behind
`RETRIEVAL_MODE`; **default stays `vector`**. Gemini stays the answer and embedding model. No LangGraph.

## 1. Evidence, trade-offs and open risks
* **Gemini space (production embeddings), 42 frozen questions, fixed parameters** (addendum): recall@5 0.564 → 0.596,
  MRR 0.415 → 0.468; held-out recall@5 0.576 → 0.576, MRR 0.390 → 0.528; dataset 0.67 → 0.67. All acceptance criteria
  passed **narrowly**: per question 5 better / 5 worse on recall@5 (17 / 8 on MRR). The gain is mostly ranking.
* **Explicit trade-offs:** modest overall gain (+0.03 recall@5); **page recall@5 regresses 0.699 → 0.647** and facet
  coverage 0.697 → 0.671; retrieval latency 3 ms → 24 ms p50 / 40 ms p95 (local).
* nomic-space results (larger gains) and AI-judged answer scores are preliminary supporting evidence only. The agentic
  graph is out of scope; its correctness advantage over BM25 one-shot was not established by this evaluation.
* **Unresolved risk — small libraries:** with per-user statistics, a user with 1–2 papers (~100 chunks) gets noisy
  IDF and a weak lexical signal. Behaviour is tested for correctness (no errors, sane fallback) but its **retrieval
  quality is unmeasured**.

## 2. Statistics: per-user, computed at query time
Stored per chunk (pure functions of the chunk's own text): `content_tsv = to_tsvector('english', content)` and
`token_count = Σ positions` (BM25's |d|). N, avgdl and df are computed **inside the query over the requesting user's
chunks only** — nothing to refresh, nothing to go stale, and no other tenant can influence a user's ranking. For a
single-user corpus this equals the evaluation harness exactly (verified by an equivalence check, §6).
`document_ids` (optional) narrows candidates, never the statistics.

## 3. Staged migration (`scripts/migrations/bm25.py up|down|status`)
Measured on PostgreSQL 16: a STORED generated column rewrites the table under `AccessExclusiveLock` (2.6 s / 20k
rows); the staged path below takes no long lock. Every DDL runs with `SET lock_timeout = '5s'` (fail fast, retry later).
1. `documind_token_count(tsvector)` (IMMUTABLE SQL function).
2. `ALTER TABLE document_chunks ADD COLUMN content_tsv tsvector, ADD COLUMN token_count integer` — nullable, no
   default: metadata-only, no rewrite.
3. `BEFORE INSERT OR UPDATE OF content` trigger fills both columns, so new and edited chunks are always current.
4. Batched backfill of rows where `content_tsv IS NULL` (1,000 rows per short transaction).
5. `CREATE INDEX CONCURRENTLY` GIN on `content_tsv` (outside a transaction; an INVALID index left by a failure is
   dropped and rebuilt).
6. Verify zero NULLs, `ANALYZE`, record the migration.
**Down** (reverse order): `DROP INDEX CONCURRENTLY`, drop trigger and functions, `DROP COLUMN` (metadata-only).
Embeddings and existing columns are never touched. **Rollback order in production:** set `RETRIEVAL_MODE=vector`
(instant) → run `down` only if needed. The runner refuses any non-loopback database without a separate
production approval.

## 4. Lifecycle
| Event | Behaviour |
|---|---|
| Upload/ingest | trigger fills columns on INSERT; chunks become visible to statistics at the ingest commit |
| Update of `content` | trigger recomputes |
| Delete | ORM cascade removes chunks; statistics change at commit |
| Failed ingestion | rollback → no rows, statistics unchanged |
| Re-ingestion | not a feature today; would be delete + insert in one transaction |

## 5. Flag, fallback, observability
* `RETRIEVAL_MODE=vector|hybrid_bm25` (default `vector`; unknown value or missing migration in hybrid mode → startup error).
* Vector mode runs the **unchanged legacy query**.
* Hybrid runs as one SQL statement inside a SAVEPOINT with `SET LOCAL statement_timeout` (default 300 ms). On
  timeout (`57014`) or any database error: `ROLLBACK TO SAVEPOINT`, then the legacy vector query. A question with no
  searchable terms runs hybrid with an empty lexical list (identical to vector order).
* One structured log line per fallback — `event=retrieval_fallback reason=timeout|error|no_lexemes mode=hybrid_bm25
  latency_ms=…` — never the question or document text; in-process per-reason counters.

## 6. Acceptance before requesting production approval
1. Existing API tests and retrieval tests pass; vector mode returns exactly the legacy results.
2. Staging copy of the evaluation DB: production hybrid top-5 ids **and** scores equal the evaluation harness for all
   42 questions; vector mode equals System A's retrieval exactly.
3. Fixed-parameter benchmark reproduces §1 (recall@5, MRR, page recall, dataset, p50/p95).
4. Zero isolation failures (another user's chunks neither appear nor change scores).
5. Migration rehearsed up → down → up on a local database with vector parity after each step.

## 7. Latency, cost, risks
Local p50 24 ms / p95 40 ms (1,003 chunks); query-time statistics scan the user's chunks (O(user chunks)); Supabase
network/CPU unmeasured. No new services or dependencies; storage ≈ 1–2× chunk text. Risks: small-library quality
(unmeasured), English-only stemming, very large accounts (fallback design: trigger-maintained per-user statistics),
the pre-existing NUL-byte ingestion bug (F-3), Supabase pauses.

## 8. Local/staging implementation evidence (2026-10-16)
Code: `api/core/retrieval.py` (flag, hybrid SQL, savepoint fallback, logs/counters), `api/routers/chat.py` (calls
`retrieve_chunks`), `api/main.py` (startup validation), `scripts/migrations/bm25.py` (staged up/down/status, local-only
guard), tests `api/tests/test_retrieval.py`; staging runner `evaluation/staging_bm25.py`. Staging DB
`documind_eval_staging` = copy of the Gemini evaluation DB (the original was not migrated). Outputs:
`evaluation/results/runs/staging_bm25_20261016T031809Z/`.
* **Tests:** API 41 passed (27 existing + 14 new), worker 4, evaluation 151. New tests cover vector-mode parity with
  the legacy query, BM25+RRF equal to an independent Python implementation, isolation of candidates *and* statistics,
  the document filter, empty / one-chunk / two-chunk libraries, no-lexeme queries, timeout (pg_sleep under a 50 ms
  local timeout) and error fallbacks with savepoint recovery and content-free logs, trigger lifecycle (insert, update,
  delete, rolled-back ingestion), migration up/down/up, idempotency, INVALID-index rebuild, startup validation, and
  `/chat` end to end in hybrid mode.
* **Equivalence (42 questions, Gemini query vectors from cache, 0 API calls):** production vector top-5 = System A
  retrieval for 42/42; production hybrid top-5 ids, per-list ranks and RRF scores = evaluation harness for 42/42
  (tolerance 1e-12); 0 fallbacks.
* **Benchmark (production code path, top-5 lists):** recall@5 0.564 → 0.596; MRR 0.393 → 0.444; page recall@5
  0.699 → 0.647; dataset recall@5 0.667 → 0.667; held-out recall@5 0.576 → 0.576, MRR 0.362 → 0.515. Latency p50/p95
  vector 3.0/4.2 ms, hybrid 24.4/36.1 ms (local Docker). (MRR here scores the top-5 list actually used; §1's figures
  score a top-10 list.)
* **Isolation:** a second user with 120 near-copies of user 1's chunks plus generic terms: user 1's rankings and
  scores unchanged for 42/42, no cross-user chunks either way; synthetic user removed.
* **Migration rehearsal on staging:** up (backfill 1,003 rows 0.30 s, index valid, 0 NULLs) → down (original 4 columns,
  vector results identical 42/42, hybrid startup refused) → up (equivalence again 42/42, 0 fallbacks).
* **Default:** `RETRIEVAL_MODE` unset → `vector` (asserted in the runs and tests).
* **Not done (needs separate approval):** any Supabase access, production migration, deployment, or default change.
  Small-library retrieval *quality* remains unmeasured (only correctness is tested).

---

## Addendum (2026-10-16) — Gemini-space acceptance test and risk review

### Gemini-space retrieval (`retrieval_A_20261016T030442Z.json`; re-runs `…030515Z`, `…030520Z` from cache)
Same 1,003 chunks (BM25 statistics identical to the nomic DB: 81,189 postings, 10,676 lexemes), same Gemini query
vectors for every method; parameters fixed (k1 1.2, b 0.75, RRF 60, 20 candidates). Usage: 28 `embed_query`
requests, 0 retries, 0 quota errors, 0 generation calls; re-runs made 0 API calls (dummy key) and reproduced every metric.

| Gemini space | vector | hybrid (ts_rank) | **hybrid (BM25)** | criterion |
|---|---|---|---|---|
| recall@5, all 38 | 0.564 | 0.496 | **0.596** | ≥ vector ✅ |
| MRR, all | 0.415 | 0.333 | **0.468** | ≥ vector ✅ |
| recall@5, held-out 11 | 0.576 | 0.667 | 0.576 | ≥ vector ✅ (equal) |
| MRR, held-out | 0.390 | 0.371 | **0.528** | ≥ vector ✅ |
| recall@5, dataset (n=6) | 0.67 | 0.50 | 0.67 | no regression ✅ |
| recall@10, all | 0.725 | 0.674 | **0.766** | — |
| page recall@5 / facet coverage@5 | 0.699 / 0.697 | 0.645 / 0.592 | 0.647 / 0.671 | (regressed) |
| per question vs vector, recall@5 | — | 5 better / 10 worse | **5 / 5** (sign p 1.0) | — |
| per question vs vector, MRR | — | — | **17 better / 8 worse** (sign p ≈ 0.11) | — |
| latency p50 / p95 (warm) | 2.7 / 3.0 ms | 68 / 86 ms | 24 / 40 ms | — |

By category (recall@5 vector → BM25): method 0.34 → 0.48, cross-paper 0.23 → 0.33, result 0.58 → 0.58,
limitation 1.00 → 1.00, dataset 0.67 → 0.67, lookup 1.00 → 0.67 (n = 3). BM25 worse on `octo-01`, `cross-03`,
`hpt-02`, `act-02`, `rt2-02`; better on `oxe-02`, `openvla-02`, `pi0-02`, `pi0-03`, `cross-01`.

**Verdict:** all stated criteria pass, **narrowly**. With Gemini vectors the baseline is much stronger (recall@5 0.564 vs
0.418 with nomic) and the gain shrinks to +0.03 recall@5 (a 5–5 per-question split) and +0.05 MRR; the effect is mainly
*ranking* (evidence moves up) rather than finding new evidence at top-5. Page recall@5 (−0.05) and facet coverage (−0.03)
regress — not acceptance criteria, but real. Held-out n = 11 and all questions were seen during the nomic-space diagnosis;
no significance is claimed. The first run's BM25 latency (p50 172 ms / p95 702 ms) was a cold start right after the index
build with network calls interleaved; warm re-runs are stable.

### Risk 1 — locking/backfill (measured on a throwaway local PostgreSQL 16 database, 20,000 synthetic rows)
* `ADD COLUMN … GENERATED ALWAYS AS (…) STORED` **rewrites the table** under `AccessExclusiveLock` (reads and writes
  blocked): 2.6 s for 20k rows. At today's demo scale (≈ 1k chunks) that is well under a second, but it scales with size.
* Staged alternative (measured): nullable `ADD COLUMN` without default — 2 ms, **no rewrite**; batched backfill in short
  transactions (10 × 2,000 rows) — 1.7 s total, no long lock; `CREATE INDEX CONCURRENTLY` — 1.1 s, writes not blocked.
* **Recommendation:** use the staged path (nullable columns → trigger maintaining them on INSERT/UPDATE → batched
  backfill → `CREATE INDEX CONCURRENTLY` → verify no NULLs → `VACUUM ANALYZE` → enable flag), with `SET lock_timeout`
  on every DDL so it fails fast instead of queueing behind traffic. The generated-column variant is acceptable only in a
  maintenance window at current size. Supabase's exact Postgres version was not checked (no access); this behaviour is
  the same across PostgreSQL 12–17.

### Risk 2 — per-user vs harness statistics
* The harness computes N, avgdl and df over **all** chunks in the evaluation DB, which holds one user. For that corpus,
  per-user statistics are mathematically identical, so the evaluated numbers carry over (to be proven by the planned
  equivalence test).
* For real users, statistics are computed over **that user's** chunks: scores will differ from the evaluation for
  corpora unlike the benchmark. Small libraries (1–2 papers, ~100 chunks) give noisy IDF and weaker lexical signal; this
  regime is **untested**. RRF uses ranks only, so the vector list still anchors results.
* Isolation: per-user statistics mean no other tenant's documents influence a user's ranking (corpus-wide statistics
  would). `document_ids` filters candidates, not statistics.

### Risk 3 — statement timeout and fallback (measured locally)
* `SET LOCAL statement_timeout = '300ms'` cancelled both an idle wait (`pg_sleep`) and a CPU-bound full-text join at
  0.30 s with SQLSTATE `57014` (QueryCanceled).
* A cancelled statement aborts the transaction; wrapping the hybrid query in a **SAVEPOINT** and issuing
  `ROLLBACK TO SAVEPOINT` restored the session (`statement_timeout` back to default) and the vector fallback query on the
  same transaction succeeded (12 ms). Worst case latency ≈ 300 ms + vector query. `SET LOCAL` (not session `SET`) is
  required behind Supabase's transaction pooler.
* Logging/measurement: catch `QueryCanceled` separately from other DB errors; emit one structured log line
  (`event=retrieval_fallback reason=timeout|error|no_lexemes mode=hybrid_bm25 latency_ms=…`, no question text) and a
  per-reason counter; count via log aggregation (no metrics service today). Acceptance: 0 fallbacks over the 42-question
  run on staging; alert threshold > 1 % of chat requests.
