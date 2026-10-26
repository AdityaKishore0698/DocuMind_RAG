# ResearchLens evaluation harness

(The harness predates the rename from DocuMind; database names such as `documind_eval` and the frozen
baseline prompts keep the old name on purpose, so earlier runs stay reproducible.)

Reproducible benchmark for comparing:

| System | What it is | Status |
|---|---|---|
| **A** | The current production RAG, frozen at commit `4a89dc1` | implemented (`baseline_a.py`) |
| **B** | Advanced RAG (page/section-aware ingestion, hybrid retrieval, RRF) | added in a later phase |
| **C** | Advanced RAG + bounded LangGraph workflow | added in a later phase |

Everything runs **locally** against a disposable pgvector PostgreSQL. The
harness never connects to Supabase or production, and its code and
dependencies are excluded from the production image (`.dockerignore`).

## What System A reproduces

`baseline_a.py` is a frozen copy of production at `4a89dc1`; `tests/test_fidelity.py`
re-reads `worker/tasks.py` and `api/routers/chat.py` **at that commit** (`git show`)
and asserts they still match:

| Stage | Production behaviour reproduced |
|---|---|
| Extraction | `extract_text` verbatim (PyPDF2; non-empty pages joined with `"\n"`) |
| Chunking | `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)` |
| Embeddings | `gemini-embedding-2`, `output_dimensionality=768`, no task type |
| Retrieval | pgvector cosine distance (`<=>`), top-5, filtered by `documents.user_id` |
| Prompt | identical `system_instruction`; context = top-5 chunks joined by blank lines |
| Generation | `client.aio.models.generate_content_stream`, `gemini-3.6-flash`, default config; each question is a fresh session |

Two deliberate differences — neither changes what A retrieves or answers:

1. **Batched ingestion embeddings** (quota saving). `check-embedding-equivalence`
   proves batched vectors equal the per-chunk production call; `ingest --embed-mode
   single` reproduces the exact per-chunk call pattern.
   *Pitfall handled:* for `gemini-embedding-2` the SDK merges a plain `list[str]`
   into **one** content (one aggregated embedding). The harness sends one
   `types.Content` per chunk and checks the returned count.
2. **Page provenance for scoring** (`provenance.py`). A never sees page numbers;
   the harness maps each chunk back to its PDF pages only to score retrieval.

One deliberate difference that *does* change what A ingests:

3. **NUL stripping** (`preprocess.py`, evaluation-only). PyPDF2 extracts some maths
   glyphs as NUL (0x00) characters, which PostgreSQL text cannot store
   (`ValueError: A string literal cannot contain NUL (0x00) characters`). The
   harness removes them before chunking and storage. Affected: Octo (2 NULs, 1 of 95
   chunks), Diffusion Policy (3, 2 of 114), ACT/ALOHA (3, 3 of 92). Production uses
   the same database driver and would most likely fail to ingest these three papers
   entirely; every run report lists this deviation.

## One-time setup

```bash
# 1. Isolated Python 3.11 environment (matches the production image)
uv venv --python 3.11 evaluation/.venv
uv pip install --python evaluation/.venv/bin/python -r evaluation/requirements.txt

# 2. Local pgvector PostgreSQL on 127.0.0.1:5434 (start Docker/OrbStack first)
docker run -d --name documind-eval-db \
  -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=documind_eval \
  -p 127.0.0.1:5434:5432 pgvector/pgvector:pg16

# 3. Environment (secrets come only from the environment; never commit them)
export EVAL_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5434/documind_eval
export GEMINI_API_KEY=...        # only for ingest / run / check-embedding-equivalence
```

The DB guard refuses any URL that is not on `localhost`/`127.0.0.1`/`::1`, whose
database name lacks `eval`, that mentions Supabase/pooler, or that equals `DATABASE_URL`.

## Workflow

All commands are run from the repository root with `PY=evaluation/.venv/bin/python`.

| Step | Command | Gemini calls |
|---|---|---|
| Download corpus (arXiv, title-verified, hashed) | `$PY evaluation/download_corpus.py` | 0 |
| Verify local PDFs vs `corpus.lock.json` | `$PY evaluation/download_corpus.py --verify` | 0 |
| Extract per-page text | `$PY evaluation/run_eval.py extract` | 0 |
| Check annotations (schema + quote on page) | `$PY evaluation/run_eval.py validate-annotations` | 0 |
| Predict API usage | `$PY evaluation/run_eval.py estimate [--include-drafts]` | 0 |
| Create local schema | `$PY evaluation/run_eval.py init-db` | 0 |
| Batch == single embedding check | `$PY evaluation/run_eval.py check-embedding-equivalence --samples 3` | 1 + 3 |
| Ingest corpus | `$PY evaluation/run_eval.py ingest --system A` | ≈ chunks ÷ batch size (cached afterwards) |
| Answer approved questions | `$PY evaluation/run_eval.py run --system A` | 2 per question (1 if query embedding cached) |
| Continue a stopped run | `$PY evaluation/run_eval.py run --resume <RUN_ID>` | same, only for questions without a final outcome |
| Compute metrics | `$PY evaluation/run_eval.py report <RUN_ID>` | 0 |
| Unit tests | `$PY -m pytest evaluation/tests -q` | 0 |

Only questions with `"status": "approved"` run by default (`--include-drafts`
overrides).

**Quota handling.** The Gemini free tier limits `gemini-embedding-2` by embedded
*texts* per minute per project (observed: `EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier`,
value 100), so a batch of 45 texts uses 45 units. Settings:

| Variable | Default | Meaning |
|---|---|---|
| `EVAL_EMBED_ITEMS_PER_MIN` | 90 | sliding 60 s budget of embedded texts (every attempt counts, including failed ones) |
| `EVAL_EMBED_BATCH_SIZE` | 45 | texts per batch request (clamped to the item budget) |
| `EVAL_GENERATE_RPM` | 8 | generation requests per minute |
| `EVAL_MAX_RETRIES` | 4 | retries for 429/5xx/timeouts only; each waits ≥ the server's `retryDelay` + 2 s, with exponential back-off + jitter, capped at 120 s per wait |

When the retry budget is exhausted the command stops (it never retries
indefinitely) and still writes its statistics (`results/runs/ingest_*.json` or
`run.json`) with the quota details and the remaining work. A **daily** quota
(429 whose QuotaFailure names a `PerDay` quota, or whose retry delay exceeds
10 min, e.g. `Please retry in 16h30m39s`) is never retried: the session stops on
the first such error. Observed free-tier daily limits: 1,000 embedded texts/day for
`gemini-embedding-2`, 20 generation requests/day for `gemini-3.6-flash`.

**Resuming a run.** `run --resume RUN_ID` continues a saved run across quota windows:

* Same question ids, questions-file hash, model, top-k, prompt/SQL hashes and corpus
  chunk count — any difference and the resume is refused.
* `traces.jsonl` is append-only (one record per attempt, with `session`, `attempt_seq`,
  `started_at`/`finished_at`, `outcome`, sanitized `attempt_errors`). A question's state
  is its latest record; reports score only that record.
* Outcomes: `answered` and `failed_terminal` (non-retryable error) are final;
  `incomplete` (text had streamed when the call failed), `failed_retryable` and
  never-attempted questions are retried. An incomplete answer is regenerated from
  scratch — never appended to, never counted as a success.
* `run.json` keeps every session (`sessions[]`, each with its own stats); top-level
  `stats` is their sum. A `.lock` file prevents two concurrent writers.
* Query embeddings are reused from the cache, so a resume only pays for generation. `ingest` checks the
local DB for inconsistent or partial state before resuming, reuses completed
papers and cached embeddings, and inserts each paper in a single transaction.
`run` refuses to benchmark an incompletely ingested corpus.

### Human review (answer correctness)

**Cross-system review (preferred):** `evaluation/human_review.py` builds one blinded CSV with every system's
answer, context and citations side by side, validates scores and summarises them — see
[HUMAN_REVIEW.md](HUMAN_REVIEW.md) for the scoring guide and commands. The per-run sheet below still works.


`report` writes `results/runs/<RUN_ID>/review.csv`. Score each answer, save the
sheet as `human_scores.csv` in the same folder, and re-run `report`:

* `correctness`: **2** = correct and complete per the gold evidence, **1** = partially
  correct / missing a key fact, **0** = wrong, unsupported, or a wrong refusal
  (negative questions: 2 = correctly says the papers don't answer it).
* `paper_attribution` (cross-paper questions): **1** = facts attributed to the right papers, **0** otherwise.
* A question may carry a `scoring_rule` (copied into `review.csv`) that overrides the
  generic rubric — e.g. `crossformer-01`, where the paper itself states two different
  embodiment counts.

### Annotation conventions

* Every material claim in a `gold_answer` must be backed by an `evidence` quote copied
  from the extracted PDF text of the stated page (`validate-annotations` checks this).
* `alternatives` list other pages that state the **same complete** fact; a passage that
  states only part of the claim is not accepted as an alternative.
* Quotes keep the paper's own spelling. Matching normalizes ligatures, case, whitespace
  and word-internal hyphens (PDF line breaks split words as `open- source`).

## Metrics

All metrics are deterministic functions in `metrics.py` (unit-tested). A metric
that cannot be computed for a question is `None` and excluded from averages;
reports show `n` so "not measured" never appears as 0.

| Metric | Definition |
|---|---|
| `evidence_recall@k` | share of gold evidence quotes found in a top-k chunk of the right paper (normalized substring, or ≥ 80 % token coverage when a quote straddles a chunk boundary) |
| `page_recall@k` | share of gold (paper, page) pairs overlapped by a top-k chunk |
| `evidence_precision@k` / `page_precision@k` | relevant chunks ÷ chunks returned |
| `evidence_mrr` / `page_mrr` | reciprocal rank of the first relevant chunk |
| `paper_recall@k` | share of the question's papers present in the top-k (multi-paper attribution at retrieval time) |
| `facet_coverage@k` | evidence coverage: share of gold facets (method, dataset, result, …) covered |
| `keyword_recall` | *heuristic*: expected key facts present in the answer |
| `negative_abstention_rate` / `false_abstention_rate` | *heuristic*: refusal detected on negative / answerable questions |
| `human_correctness`, `human_paper_attribution` | from the human rubric (above) |
| latency p50/p95 | query embedding, DB retrieval, time-to-first-token, generation, end-to-end (pacing sleeps excluded) |
| Gemini usage | requests and attempts by kind, retries, first-attempt and final errors, generation tokens from `usage_metadata` (prompt / output / thinking / total) |

**Not applicable to System A:** citation support — A is not prompted to cite and
returns no source ids. It becomes measurable for Systems B/C.

## Files

| Path | Committed? | Purpose |
|---|---|---|
| `corpus.yaml` | yes | paper list (arXiv id, version, title, licence, enabled flag) |
| `corpus.lock.json` | yes | sha256 / size / pages of each downloaded PDF |
| `questions.jsonl` | yes | questions + gold evidence (short quotes, page numbers) |
| `requirements.in` / `requirements.txt` | yes | harness deps / pinned lock (Python 3.11) |
| `data/` | **no** | PDFs, extracted pages |
| `results/runs/` | **no** | raw traces (contain retrieved paper excerpts) |
| `results/summaries/` | optional | metrics-only JSON/Markdown, safe to commit |

## Licensing

PDFs are fetched from arXiv and never redistributed (most corpus papers use
arXiv's non-exclusive distribution licence; three are CC BY 4.0). Gold
annotations quote at most a short sentence per evidence item.
