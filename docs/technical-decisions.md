# Technical decisions

Each entry: decision · alternatives · why · trade-off. Only decisions actually taken are listed.

### TD-1 Freeze the baseline as a code copy, verified against git history
* **Decision:** `evaluation/baseline_a.py` copies production's constants, prompt, SQL and
  `extract_text` from commit `4a89dc1`; `tests/test_fidelity.py` re-reads those files via `git show`
  and asserts equality.
* **Alternatives:** benchmark the live API; import production modules directly.
* **Why:** later phases will change production code; a baseline must not drift, and benchmarking
  production would touch real data and quotas.
* **Trade-off:** duplication — mitigated by the fidelity test.

### TD-2 Local, guarded evaluation database
* **Decision:** Docker pgvector on `127.0.0.1:5434`; `db.assert_local_eval_database` rejects
  non-local hosts, names without `eval`, Supabase/pooler URLs, and `DATABASE_URL` itself.
* **Why:** zero risk to production data; reproducible. **Trade-off:** must keep the container alive
  (an OrbStack stop once took the DB offline — see failure-analysis F-6).

### TD-3 Batched embeddings during evaluation ingestion
* **Decision:** send up to 45 `types.Content` objects per request instead of production's one call
  per chunk. **Verified** equivalent: batched vs single vectors cosine 1.000000 (`check-embedding-equivalence`).
* **Why:** fewer requests. **Trade-off:** differs from production's call pattern (recorded as a
  deviation; `--embed-mode single` reproduces it).

### TD-4 Pace embeddings by *items*, not requests
* **Decision:** sliding 60 s window of embedded texts (default 90/min, batch 45).
* **Why:** the free-tier limit counted texts (100/min) — request-based pacing still hit 429s.

### TD-5 Bounded retries, server delay respected, daily quota never retried
* **Decision:** retry only 429/5xx/timeouts, max 4, wait ≥ server `retryDelay` + 2 s, exponential
  back-off with jitter, ≤ 120 s per wait. A 429 for a per-day quota (or delay > 10 min) stops at once.
* **Why:** retrying a quota that resets in 16 h wastes ~8 min and adds rejected calls.

### TD-6 Strip NUL characters (evaluation-only), outside the frozen baseline
* **Decision:** `preprocess.strip_nul` after extraction; documented as a deviation in every run.
* **Alternatives:** exclude the 3 affected papers; replace NUL with a space; fix production now.
* **Why:** smallest change that lets the corpus ingest; production fix belongs in Phase 3.
* **Trade-off:** System A is measured slightly *better* than production, which would most likely
  fail to ingest those papers at all.

### TD-7 Append-only, resumable benchmark runs
* **Decision:** `traces.jsonl` is an append-only attempt log; a question's state is its latest
  record; `run --resume` re-attempts only non-terminal questions under an identical configuration.
* **Alternatives:** start fresh each day (impossible: 42 questions > 20 generations/day); paid tier.
* **Why:** keep the free tier, never lose or overwrite results. **Trade-off:** answers are generated
  on different days (model-side drift possible); recorded per record via `started_at` and `session`.

### TD-8 Partial streamed answers are failures, regenerated from scratch
* **Decision:** if text has streamed when a call fails, the attempt is `incomplete`, not retried
  in-session, and regenerated in full on resume. **Why:** concatenating two generations would score
  an answer the model never produced.

## Local-inference branch (`feature/local-inference-qwen3-14b`)

### TD-9 One provider boundary, Gemini default, no silent fallback
* **Decision:** `api/core/inference.py` (`GeminiProvider`, `OllamaProvider`, `get_provider()`) selected by
  `INFERENCE_PROVIDER`; `worker/inference.py` mirrors the embedding half. Routes call `embed()` /
  `stream_chat()` only.
* **Alternatives:** LangChain/LiteLLM abstraction; Ollama's OpenAI-compatible endpoint; a separate
  inference microservice. **Why not:** two call sites and two providers don't justify a framework or a
  new service; the native `/api/chat` exposes Ollama-specific controls (`think`, `keep_alive`, `num_ctx`).
* **No fallback:** silently answering with a hosted model would invalidate local-vs-hosted comparisons
  and send documents to a third party the operator opted out of.

### TD-10 Embedding spaces are kept physically separate
* `nomic-embed-text` and Gemini both emit 768-d vectors, so a mixed table would *work* and silently return
  meaningless neighbours. The Ollama provider refuses any non-loopback `DATABASE_URL`; the local smoke
  test requires a DB named `*local*` and not `*eval*`. Switching providers on real data means re-embedding.

### TD-11 Reuse the lab server's existing Ollama via an SSH tunnel
* **Decision:** use the system Ollama service (already running, loopback-bound, `nomic-embed-text`
  present) through `ssh -L 127.0.0.1:11435:127.0.0.1:11434` with a dedicated control socket.
* **Alternatives:** a second, user-level Ollama daemon (duplicate model store, competes for VRAM);
  binding Ollama to 0.0.0.0 (unauthenticated API on the network). Both rejected.
* `keep_alive=2m` (vs Ollama's default 5m) releases VRAM sooner on a shared GPU, at the cost of reloads.

### TD-12 Model choices (as specified) and their constraints
* `qwen3:14b` Q4_K_M: 9.28 GB file (pulled 2026-10-15, digest `bdbd181c33f2`). `think=false` by default:
  Qwen3's reasoning mode adds hidden tokens and latency; configurable for comparison.
* `nomic-embed-text` v1: 2,048-token context (verified metadata), trained with task prefixes, so the
  provider adds `search_query: ` / `search_document: `.

### TD-13 Smaller generation model on the Mac; model choice lives in env files, not code
* **Decision:** Mac development uses `qwen3:8b`, the lab GPU `qwen3:14b`; `OLLAMA_CHAT_MODEL` is required
  (default removed) and each environment has its own template.
* **Why:** the Mac (M2) has 16 GB of unified memory shared by the OS, Docker/Postgres and the model;
  qwen3:14b needs ~11 GB (weights + 8k KV cache, computed). Measured on the Mac: qwen3:8b resident 5.63 GB;
  with nomic loaded too, free memory fell to 20 % and macOS reported pressure level 2 (warn) — 14B would not fit.
* **Trade-off:** Mac answers are not comparable to lab answers — use the Mac for integration/debugging,
  the lab model for any quality measurement. Embeddings are identical (`nomic-embed-text`) in both.

## Evaluation without the Gemini quota; Phase 2 retrieval

### TD-14 Local-model baseline (System L) as a separate system, not a variant of Baseline A
* **Decision:** System `L` = System A's frozen pipeline (extraction, NUL strip, chunking, retrieval SQL, prompt,
  top-5) with only the two model calls swapped: `nomic-embed-text` (with nomic task prefixes) and `qwen3:8b`
  (think=false, num_ctx 8192) via the Mac's Ollama (`evaluation/ollama_backend.py`).
* **Isolation:** its own database (`documind_eval_ollama` on the 5436 container) and run ids `*_L`; a guard
  (`assert_single_embedding_space`) refuses to ingest/run a system in a DB holding another system's vectors.
  The Gemini run `20261015T072037Z_A` is untouched.
* **Why separate:** both the embedding space and the generator differ, so L's numbers are a *different baseline*,
  not a substitute for A. (Baseline A was later stopped at 13/42 — failure-analysis F-15 — so the complete
  comparisons in this project are the local ones.) **Trade-off:** answers are not comparable to Gemini's; retrieval comparisons are only
  valid within one embedding space.

### TD-15 Hybrid retrieval measured on the same chunks before touching ingestion or production
* **Decision:** implement vector + PostgreSQL FTS + RRF (`evaluation/hybrid.py`) and evaluate it with
  `run_eval.py retrieval-eval` — same chunks, same query embeddings, no generation — so the only variable is
  the retrieval method. Production schema (stored tsvector + GIN index) is deferred until the design is settled.
* **FTS design:** question lexemes OR-ed (AND would rarely match a whole question), cast directly to tsquery
  (keeps `x-embodi` as one lexeme), ranked by `ts_rank_cd`; RRF k=60, 20 candidates per list.
* **Measured result:** see evaluation.md — this first design regresses top-5 and improves top-10 (open decision).

### TD-16 BM25 lexical ranking inside PostgreSQL (no external search engine)
* **Decision:** keep PostgreSQL FTS for matching (`@@`, GIN expression index) but rank with Okapi BM25
  (k1 = 1.2, b = 0.75, Lucene's non-negative IDF), using statistics materialised from the same `to_tsvector`
  (`eval_bm25_postings` tf, `eval_bm25_doclen`, `eval_bm25_stats` N/avgdl + corpus fingerprint).
* **Alternatives:** document-frequency cut-off (a tuned threshold), weighted RRF (tuned weight), an extension
  (pg_search/ParadeDB — availability on Supabase unverified), Elasticsearch/OpenSearch (excluded by the plan).
* **Overfitting safeguards:** parameters fixed a priori; paper-group split registered before measuring; no
  parameter changes after seeing held-out results.
* **Consequences for production (not done):** needs a migration (GIN index; a stored tsvector or statistics tables)
  and a statistics refresh on every ingest/delete — the harness rebuilds after ingestion and refuses stale stats.

### TD-17 Phase 3: one bounded LangGraph workflow, evaluated before it touches the app
* **Graph:** plan (≤ 3 subqueries, JSON) → retrieve (per-query rankings fused with RRF) → assess (JSON:
  sufficient? follow-up) → [≤ 2 retrieval rounds, 300 s budget, repeated query ends loop] → synthesize (top-5
  fused chunks labelled `[S#] (filename)`, cite every claim) → verify (deterministic citation check).
* **Why this shape:** the plan's single-workflow, bounded design; one planning call and one assessment per round
  keep cost predictable; deterministic verification instead of an LLM judge.
* **Controls:** equal 5-chunk context budget for every system; CLV vs L isolates orchestration (both vector),
  CL vs BL isolates orchestration on BM25 hybrid, BL vs L isolates retrieval.
* **Alternatives not taken:** multiple agents, tool-calling ReAct loops, rerankers, an LLM-as-judge — more
  infrastructure and less control than this comparison needs.
* **Known confound:** the graph's synthesis prompt adds source filenames and a citation instruction that the
  frozen one-shot prompt lacks; differences are attributable to "orchestration + cited prompt", not the graph alone.
* **Observed before the comparison (3-question smoke, not tuned):** qwen3:8b's assessment answered "insufficient"
  every time and proposed paraphrases of the question as follow-ups, so most questions use 2 rounds.
* **Outcome (measured, preliminary):** the graph did not improve the retrieved context (recall@5 CL 0.522 vs
  BL 0.544; CLV 0.398 vs L 0.418) and costs ~2.3× tokens and ~3× TTFT; answer keyword coverage rose modestly
  (+0.04 / +0.06, not significant, prompt-confounded). Decision deferred to human correctness scoring; BM25
  hybrid (BL) is the stronger candidate for the application.

### TD-18 AI judging instead of manual scoring — scoped, blinded, two-pass
* **Decision:** Claude Opus subagents (Claude Code plan; no API keys) score only the 44 priority rows (cross-paper +
  negative) with the unchanged rubric; a second fresh-subagent pass on the same rows; pass 1 primary, disagreements
  flagged; un-blinding only after both passes were saved (`llm_judge.py` → `human_review.py summarize --key`).
* **Why scoped:** a pilot showed ~40k tokens of per-subagent overhead (64k for 12 rows vs ~35–40k estimated); judging
  all 168 rows twice would have cost ~1.1M tokens. The 44 rows answer the open Phase 3 question.
* **Alternatives:** Gemini via API (quota exhausted, ruled out), Antigravity CLI (not installed; plan quota unknown),
  local llama3 8B (weaker than the generator), qwen3:14b on the lab GPU (saturated; same family as the generator).

### TD-19 BM25 in the application: flag-gated, per-user query-time statistics, staged migration
* `RETRIEVAL_MODE=vector|hybrid_bm25` (default vector; unknown value or missing migration → startup error); vector mode
  runs the unchanged legacy ORM query. Hybrid = one SQL statement (per-user N/avgdl/df from trigger-maintained
  `content_tsv`/`token_count`, vector top-20, BM25 top-20, RRF) inside a SAVEPOINT with a local 300 ms
  `statement_timeout`; timeout/error → rollback to savepoint → legacy vector query; content-free log + counter.
* **Why:** exact equivalence with the evaluated configuration, no cross-tenant statistics, nothing to refresh, no
  long table locks (measured), instant rollback by flag.
* **Validated locally only** (proposal §8); production migration/deploy not approved.

## Research-paper analyst in the application (plan Phases 1–5)

### TD-20 Migrate in place instead of the plan's clean reset
* **Decision:** additive migration `002_provenance` (nullable columns, concurrent indexes, FK swapped via `NOT VALID` +
  `VALIDATE`, short `lock_timeout`, idempotent, `down` provided) plus a startup schema check, instead of wiping
  production. `init_db.sql` matches the migrated schema for new databases.
* **Why:** keeps users' data, needs no maintenance window, and old code keeps working on the new schema, so code
  rollback never requires a schema rollback. Old documents simply lack provenance until re-uploaded.

### TD-21 Page/section-aware chunking with the same splitter
* **Decision:** keep `RecursiveCharacterTextSplitter(1000, 100)` but apply it within page and section spans; store
  page, section and position per chunk; detect headings with conservative rules (canonical names, small caps,
  run-in, numbered paper-specific titles) rather than a layout model.
* **Why:** page numbers are what citations need; the splitter stays the evaluated one, so retrieval differences
  come from boundaries only (measured as System PL, see evaluation.md). A layout parser (GROBID, Nougat) would be
  more accurate but adds a heavy service.

### TD-22 Citations are numbered sources chosen before generation
* **Decision:** `[S#]` labels on retrieved chunks, sent to the client in `X-Sources` before the first token and
  stored with the message with the parsed markers; invalid markers are recorded, not hidden.
* **Why:** streaming stays plain text; the client can render sources immediately; code can check that a marker
  names a supplied source (support itself is not machine-verified — stated in the UI).

### TD-23 The agent ships behind a flag, with deterministic coverage and balanced context
* **Decision:** port the evaluated graph (TD-17) without its synthesis/verify nodes — the endpoint streams the
  synthesis and checks citations — and add a deterministic coverage step plus round-robin context slots per
  selected paper; time budget 60 s; any graph error falls back to plain RAG. `AGENT_ENABLED=false` by default.
* **Why:** TD-17 measured no retrieval gain at ~2–3× tokens and latency, so it is opt-in. Coverage and balanced
  slots address the failure the evaluation exposed for cross-paper questions (one paper crowding out the other):
  in the local end-to-end run, the first version gave 4 of 5 slots to one paper; round-robin fixed it.

### TD-24 Structured analysis = one JSON call per paper, validated in code
* **Decision:** comparisons call the model once per paper with that paper's sources only, then validate: unknown
  or missing citations turn the value into "not reported in the paper". Gaps use one call over all papers and
  drop uncited gaps.
* **Why:** per-paper calls make cross-paper attribution errors impossible by construction and let one paper fail
  without sinking the table; validation in code is cheaper and more predictable than an LLM verifier.

### TD-25 Gap scope is decided by citations, not by the model
* **Decision:** the gap prompt groups evidence by paper and asks for cross-paper gaps first, but the label shown to
  users comes from code: a gap whose valid citations span ≥ 2 papers is a *cross-paper hypothesis*, one paper is a
  *single-paper observation*. One bounded follow-up call (only when nothing crossed papers and ≥ 2 papers had
  evidence) asks the model to link the single-paper observations. Caps 5 + 3.
* **Why:** the local run (F-19) showed the model calling restated single-paper limitations "gaps"; a
  model-written label cannot be trusted, citations can be checked. The follow-up costs one extra call only in the
  failure case. Whether a cited cross-paper gap is *meaningful* is still not machine-checked.

### TD-26 Papers belong to the chat they were uploaded in
* **Decision:** `documents.session_id` (nullable FK to `chat_sessions`, ON DELETE CASCADE, migration 004); one
  paper belongs to exactly one chat. Every paper endpoint and retrieval query filters on user **and** chat
  (`session_id` is a required keyword argument of the retrieval functions, so a new call site cannot forget it).
  Deleting a chat deletes its papers and stored files.
* **Legacy papers:** no backfill. Rows created before the migration keep `session_id = NULL`, are searched in no
  chat, and are listed separately so the user can move each into one chosen chat. Guessing a chat (newest, first)
  would silently mix papers into conversations they were never part of.
* **Alternatives not taken:** a many-to-many chat↔paper table (sharing one paper across chats). It is more
  flexible, but adds a join to every query and makes "deleting a chat" ambiguous. Re-uploading a paper in another
  chat is cheap with local embeddings.
* **Trade-off:** BM25 statistics are now computed per chat rather than per user (smaller corpus, same formula).
