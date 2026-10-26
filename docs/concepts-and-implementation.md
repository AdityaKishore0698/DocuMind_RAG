# Concepts and implementation

Only implemented components are described. Planned concepts (hybrid search, RRF, LangGraph) are
added when they are built.

## RAG as implemented in System A
* **Chunking** — `RecursiveCharacterTextSplitter(1000, 100)` splits on paragraph → line → word
  boundaries until pieces fit 1,000 characters; 100-character overlap keeps sentences that straddle a
  boundary retrievable from both sides. Character-based, not token- or section-aware.
* **Embeddings** — `gemini-embedding-2` with `output_dimensionality=768` (Matryoshka-style truncation
  to fit `Vector(768)`); no `task_type`, so queries and documents share one embedding space setting.
* **Vector search** — pgvector `<=>` is cosine *distance* (1 − cosine similarity); `ORDER BY … LIMIT 5`
  without an index is an exact scan (fine at ~1k chunks; would need HNSW/IVFFlat at scale).
* **Grounding** — the top-5 chunks are placed in `system_instruction` with "if it does not contain the
  answer, say so". No citations, no page numbers: the model cannot point to sources.
* **Streaming** — async Gemini stream → FastAPI `StreamingResponse`; time-to-first-token (TTFT) is the
  user-perceived latency.

## Evaluation harness internals (`evaluation/`)
* **Page provenance** (`provenance.py`) — rebuilds production's concatenated text while recording each
  page's character span, then locates every chunk in that text to map it to PDF pages. Production
  never sees pages; the mapping exists only to score retrieval against page-level gold evidence.
* **Embedding batching pitfall** (`gemini.py`) — with google-genai 2.30.1, `contents=list[str]` is
  merged into ONE content → one aggregated vector. The harness sends one `types.Content` per text and
  checks the returned count, falling back to single calls on mismatch.
* **Embedding cache** (`db.py`) — keyed by (model, dim, sha256(text)); makes ingestion and query
  embedding resumable and idempotent. pgvector returns a `Vector` object → `vector_to_list`.
* **Rate limiting** — `ItemPacer` (sliding window of items), `Pacer` (min interval per request),
  `retry_wait_seconds` (max(back-off, server delay + 2 s), cap 120 s), `classify_error`
  (`rate_limit` / `daily_quota` / `server` / `timeout` / `client_<code>`), `parse_duration`
  (`16h30m39.2s` → seconds).
* **Error logging** — `error_details` keeps type, HTTP status, Google status and the provider message
  with API keys/bearer tokens redacted and length capped; prompts and paper text are never logged.
* **Resumable runs** (`run_eval.py`) — append-only `traces.jsonl`; outcome per record (`answered`,
  `incomplete`, `failed_retryable`, `failed_terminal`); `pending_question_ids` selects work;
  `resume_mismatches` refuses config/corpus/question drift; `.lock` prevents concurrent writers;
  `merge_stats` sums per-session usage.

## Evaluation concepts
* **Evidence recall@k** — share of gold evidence quotes found in a top-k chunk (normalized substring,
  or ≥ 80 % token coverage across a chunk boundary). **Precision@k** — relevant chunks ÷ k.
* **MRR** — 1 / rank of the first relevant chunk, averaged; rewards putting evidence first.
* **Facet coverage** — share of a question's gold facets (method, dataset, result…) retrieved.
* **Latency percentiles** — p50/p95 with linear interpolation; pacing and back-off sleeps excluded.
* **Undefined ≠ zero** — a metric that cannot be computed is `None`, excluded from means, and `n` is
  reported, so "not measured" never looks like "scored 0".

## Local inference (branch `feature/local-inference-qwen3-14b`)
* **Ollama** — a llama.cpp-based model server. `/api/embed` returns `{"embeddings": [[…]]}`;
  `/api/chat` streams NDJSON events (`message.content` deltas, a final `done: true` with token counts);
  `/api/show` returns GGUF metadata without loading the model; `/api/ps` lists loaded models and VRAM.
* **Quantization** — Q4_K_M stores weights in ~4–5 bits with per-block scales ("K-quants"), cutting a
  14.8B-parameter model from ~30 GB (FP16) to ~9.3 GB at a small quality cost. `nomic-embed-text` is
  served unquantized (F16, 137M parameters).
* **GPU memory = weights + KV cache + runtime buffers.** KV cache grows linearly with `num_ctx`.
  For qwen3-14B the architecture (40 layers, 8 KV heads × 128 dims — verified from GGUF metadata) gives, at
  FP16, 2 × 40 × 8 × 128 × 2 B ≈ 160 KB per token → ≈ 1.3 GB at 8,192 tokens. *Computed, not measured.* If it doesn't fit, Ollama offloads layers
  to CPU (much slower).
* **Embedding dimension ≠ embedding space** — equal length (768) only means the vectors fit the column;
  similarity is meaningful only between vectors from the same model (and, for nomic, the right prefixes).
* **Implementation** — `OllamaProvider.embed` checks count and dimension; `stream_chat` maps history roles,
  surfaces `error` events and HTTP errors as `InferenceUnavailable`; errors name the endpoint and exception
  class, never the request body. httpx `trust_env=False` keeps the private tunnel off any system proxy.
* **Environment separation** — `ollama_settings()` reads endpoint, models, timeouts, `keep_alive`, `num_ctx`,
  `think` and prefixes from the environment; `OLLAMA_CHAT_MODEL` has no default. Templates
  `local_inference/env.example` (lab, qwen3:14b, tunnel port 11435) and `env.mac.example` (Mac, qwen3:8b,
  port 11434) differ only in endpoint and chat model; a unit test loads each and asserts the selection.

## Hybrid retrieval (Phase 2, evaluation harness — not yet in the application)
* **PostgreSQL full-text search** — `to_tsvector('english', text)` normalises words to stemmed lexemes
  (`embodiment` → `embodi`, stop words dropped); `tsquery` expresses which lexemes must match (`&`, `|`, `<->`);
  `@@` tests a match; `ts_rank_cd` scores by cover density (how many query terms appear and how close together).
  **It has no IDF:** a term in 66 % of chunks counts as much as a rare one — the cause of the regression below.
* **Reciprocal Rank Fusion** — `score(d) = Σ 1/(k + rank_i(d))`, k = 60. Uses ranks only, so incomparable raw
  scores (cosine distance vs ts_rank) never need normalising; documents found by both lists rise to the top.
* **Implementation** (`evaluation/hybrid.py`): two candidate queries (vector top-20, FTS top-20) scoped by
  `user_id` and optional `document_ids`; `rrf_fuse` (pure, deterministic tie-break); results keep `vector_rank`,
  `fts_rank`, `rrf_score` and page provenance. Query-time `to_tsvector` costs ~0.27 s per query on 1,003 chunks
  (no index) vs 5 ms for exact vector search.
* **BM25 vs ts_rank** — BM25 weights each term by IDF and saturates term frequency (k1, b); it is the usual lexical
  half of hybrid search. Postgres core lacks it; options are an IDF table built from `ts_stat`, an extension, or
  filtering corpus-frequent lexemes.
* **BM25 implementation** (`hybrid.py`): `build_lexical_index` creates the GIN index and fills postings
  (lexeme, chunk, tf from tsvector positions), doc lengths and N/avgdl in one transaction, plus a fingerprint of
  the chunk table; `BM25_SQL` matches with `@@` and sums IDF·saturated-tf over the query's lexemes. Ingestion
  rebuilds the statistics; `assert_lexical_index_fresh` refuses stale ones. Test: scores equal an independent
  Python implementation (rel 1e-9).
* **Experimental hygiene** — a development/held-out split by connected paper groups (cross-paper questions keep
  linked papers together), registered before measuring; paired per-question win/loss counts and a sign test
  instead of trusting a mean difference on 11–38 questions.

## Agentic workflow (Phase 3, `evaluation/agentic.py`)
* **LangGraph** — a `StateGraph` over a typed state (`GraphState`); nodes return partial state updates; a
  conditional edge (`route`) chooses retrieve vs synthesize after `assess`; `recursion_limit` caps node steps.
* **Bounded loop** — retrieval rounds ≤ 2, subqueries ≤ 3, wall-clock budget 300 s, and a repeated follow-up ends
  the loop; unparseable JSON from the model falls back deterministically (question only / stop looping).
* **Structured outputs** — Ollama `format` = JSON schema, temperature 0 for plan/assess (`ollama_backend.chat_json`).
* **Citations** — sources numbered `[S1..S5]`; `parse_citations` handles `[S1]`, `[S1, S2]`, `[s4]`; valid = refers
  to a provided source; cited sources map to chunk ids. Metrics: citation validity, citation support precision
  (cited chunk contains a gold quote — strict), answers with citations.
* **Dependency injection** — retrieval, JSON calls and generation are injected (`GraphDeps`), so routing and
  bounds are unit-tested without a model or DB; the harness adapter (`_answer_with_graph`) maps a graph run to
  the standard trace (final context as `retrieved`, TTFT measured from question start).
