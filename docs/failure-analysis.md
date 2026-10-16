# Failure analysis

Format: symptom · evidence · root cause · fix · regression test · lesson.

### F-1 Truncated PDF download (CrossFormer)
* **Symptom:** PDF parse failure after download. **Root cause:** connection ended early; file saved anyway.
* **Fix:** streaming download with retries; `check_pdf_complete` (Content-Length, `%%EOF`, PyPDF2 parse).
* **Test:** `tests/test_download.py`. **Lesson:** validate artefacts at the boundary, not at use.

### F-2 Embedding 429s during ingestion (per-minute quota)
* **Evidence:** `quotaId EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier`, value 100,
  hit while sending ~2 batches/min — i.e. the quota counts *texts*, not HTTP requests (inference
  from the observed numbers).
* **Root cause:** request-based pacing; SDK default is no retry.
* **Fix:** `ItemPacer` (90 texts/min), retry honouring `RetryInfo.retryDelay`, stats persisted on failure,
  per-paper transactional inserts for safe resume. **Tests:** `test_gemini.py` (fake clock, quota parsing).
* **Lesson:** read the quota id; "requests per minute" can mean items.

### F-3 NUL bytes in extracted PDF text
* **Symptom:** `ValueError: A string literal cannot contain NUL (0x00) characters` on insert (Octo).
* **Root cause:** PyPDF2 extracts some math glyphs as `\x00`; PostgreSQL `text` cannot store NUL.
  Affects Octo (2), Diffusion Policy (3), ACT/ALOHA (3).
* **Fix:** evaluation-only `preprocess.strip_nul` (TD-6). **Not fixed in production** — production uses the
  same driver and would most likely fail to ingest these papers (unverified against the live app);
  scheduled for Phase 3.
* **Test:** `tests/test_preprocess.py` (driver rejects raw text, accepts stripped; page spans and chunk
  mapping stay consistent; real affected papers). **Lesson:** sanitize at the extraction boundary.

### F-4 Cached embeddings not iterable
* **Symptom:** `TypeError: 'Vector' object is not iterable` on resume. **Root cause:** pgvector's psycopg2
  adapter returns a `Vector` object, not a list. **Fix:** `db.vector_to_list`. Caught before any API call.
* **Test:** `test_db_guard.py` (all adapter return types + real-DB round trip).

### F-5 Daily embedding quota (1,000 texts/day)
* **Evidence:** `EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier`, value 1000, after ~960
  texts in one day. A 1-text probe succeeded while a 45-text batch failed — a small probe does not prove
  capacity for the real workload. **Resolution:** quota reset at ~00:00 PT; RT-2 ingested (47 texts).

### F-6 Local DB offline (OrbStack stopped)
* **Symptom:** `connection refused` on 127.0.0.1:5434. **Fix:** restarted OrbStack + container; data intact
  (container not removed). **Lesson:** check infrastructure before attributing failures to the API.

### F-7 Generation 503s and a partial streamed answer
* **Evidence:** `HTTP 503 UNAVAILABLE: This model is currently experiencing high demand` (5 consecutive in
  one run; later intermittent, each recovered on retry). For `openvla-03` the 503 arrived after text had
  streamed. **Initial gap:** only the exception type was logged. **Fix:** `error_details` (status, message,
  redaction); partial answers marked `incomplete` and regenerated from scratch on resume (TD-8).
* **Tests:** `test_gemini.py` (real SDK `ServerError`, redaction, no prompt in logs), `test_resume.py`.

### F-8 Daily generation quota (20 requests/day) + pointless retries
* **Evidence:** `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, value 20; message `Please retry in
  16h30m39s`. The harness retried 4× at 120 s because it could not parse `h`/`m` durations.
* **Fix:** `parse_duration`, `daily_quota` category (never retried), resumable runs (TD-7).
* **Tests:** `test_resume.py` (parsing, no-retry, stop + resume, duplicate guard, legacy run.json).
* **Lesson:** distinguish transient from window-scoped limits; design long jobs to be resumable.

## Local-inference branch

### F-9 Fresh dependency resolve breaks the database driver (SQLAlchemy 2.1)
* **Symptom:** API tests failed at import: `ModuleNotFoundError: No module named 'psycopg'`.
* **Root cause:** `api/requirements.txt` is unpinned; SQLAlchemy 2.1.4 maps `postgresql://` to psycopg 3,
  while the app installs `psycopg2-binary`.
* **Fix (branch-local):** `sqlalchemy<2.1` in `local_inference/requirements.in` (locked 2.0.54) → 24/24 API
  tests pass. **Production risk (unverified):** a fresh production image build would resolve the same way;
  production requirements were deliberately not changed. **Lesson:** lock application dependencies.

### F-10 A system HTTP proxy answered requests for 127.0.0.1
* **Symptom:** `curl http://127.0.0.1:11435/api/version` through the tunnel returned a proxy
  "Gateway Timeout" HTML page. **Root cause:** proxy environment variables applied to loopback.
* **Fix:** `curl --noproxy '*'` for checks; httpx clients use `trust_env=False`. Verified: version and
  model metadata then returned through the tunnel.

### F-11 Inference blocked by a saturated shared GPU (open)
* **Evidence (read-only, 2026-10-15 14:07, 14:16, 14:21 and 14:37 IST):** Quadro RTX 6000 at 19.0/24.6 GB used,
  100 % utilization, 86 °C, ~254 W, from other users' jobs; `qwen3:14b` not installed.
* **Decision:** no model load, no pull. ~5.5 GB free VRAM cannot hold qwen3:14b (~9.3 GB + KV cache);
  loading would offload to CPU and compete with others' work.
* **Progress:** model pull approved and completed (9.28 GB, digest verified); inference deferred, with a
  read-only 5-minute GPU poll waiting for ≥ 12 GB free VRAM, ≤ 80 % utilization and ≤ 80 °C.

### F-12 `ollama pull` hangs on the Mac's network (proxy-only egress)
* **Symptoms:** first `TLS handshake timeout` to registry.ollama.ai (launchd service has no proxy env); with
  `HTTPS_PROXY` set on a manually started server the manifest arrived but even a 420-byte blob never finished.
  Separately, the `ollama` CLI inherited the shell proxy for its own 127.0.0.1 call (same class as F-10).
* **Evidence:** `lsof` on the server showed direct connections to `104.18.17.170:443` (Cloudflare blob
  storage) alongside one connection to the proxy; curl through the proxy fetched the same blob (via the
  R2 redirect) in ~5 s; direct curl timed out.
* **Root cause (inferred):** Ollama 0.40.4's blob downloader does not route through `HTTPS_PROXY` here.
* **Fix:** fetch the registry artifacts with curl via the proxy, sha256-verify each blob, write them in the
  registry layout (manifest last). Model IDs match the registry (`qwen3:8b` 500a1f067a9f, `nomic-embed-text`
  0a109f422b47). **Lesson:** check which hosts a tool actually connects to before blaming the network.

### F-13 RAG smoke test needed numpy / mis-adapted vectors
* `check.py --rag` imported numpy (not in the app's dependency set) and passed Python lists where pgvector needs
  `vector`. The failed transaction rolled back (no rows). **Fix:** pgvector text literal + `::vector` cast;
  rerun passed (3 rows × 768-d).

### F-14 First hybrid design regresses top-5 retrieval (resolved in the harness by BM25 — preliminary)
* **Evidence** (`retrieval_L_20261015T162417Z.json`, System L chunks, 38 answerable questions): evidence
  recall@5 0.418 (vector) → 0.371 (hybrid); recall@10 0.528 → 0.575; MRR 0.290 → 0.273; paper recall@5 1.00 → 0.947.
  By type: result 0.17 → 0.50 and limitation 0.68 → 0.79 improve; dataset 0.50 → 0.00 and method 0.40 → 0.28 regress.
* **Root cause (supported by corpus statistics):** OR-queries ranked with `ts_rank_cd`, which has no IDF.
  `ts_stat` shows `robot` in 65.6 % of chunks, `task` 56.4 %, `train` 54.5 %, `model` 53.0 %; chunks dense in these
  generic terms outrank the specific evidence and displace good vector hits through RRF.
* **Validity caveat:** any fix tuned on these 42 questions risks overfitting the benchmark (no held-out split).
* **Resolution:** BM25 ranking (TD-16). Measured on the same chunks: recall@5 0.418 (vector) / 0.371 (ts_rank
  hybrid) / 0.544 (BM25 hybrid); dataset questions 0.50 / 0.00 / 0.50; held-out 0.273 / 0.417 / 0.576.
  Not significant (sign test p ≈ 0.15); production not changed. **Lesson:** a lexical scorer without IDF turns
  OR-queries into "which chunk says *robot* most often".

### F-15 Gemini Baseline A stopped at 13/42 (quota-constrained; cause unverified)
* **Evidence:** run `20261015T072037Z_A` sessions 2 and 3 each completed exactly one generation and then received
  `429 GenerateRequestsPerDayPerProjectPerModel-FreeTier` (limit 20) — session 3 at 00:05 UTC, minutes after the
  announced reset (`retry in 23h54m`). Session 1 (the first day) completed 11.
* **Cause:** not established. Hypothesis: the per-project quota is shared with another consumer; not verified
  against the provider dashboard.
* **Decision (2026-10-16):** stop the run; no new keys, billing or credential changes. The partial run is preserved
  and labelled partial; the safeguards worked as designed (immediate stop on daily quota, append-only traces,
  no duplicate answers). **Lesson:** a free-tier quota shared at project level makes a multi-day benchmark
  unreliable — budget a dedicated quota or a local model for evaluation from the start.
