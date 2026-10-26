# ResearchLens — Research Paper Analyst

ResearchLens answers questions about the research papers you upload, with a citation to the page and section
behind every claim. It compares papers side by side and suggests research gaps across them. Everything runs
on your machine: the language models are served locally by [Ollama](https://ollama.com), and auth, storage
and the vector database come from the local [Supabase](https://supabase.com/docs/guides/local-development)
stack. No paid API or cloud account is needed.

* **Cited answers.** Retrieved passages become numbered sources `[S1] … [Sn]` (paper, page, section). The
  answer cites them, and the UI lists each cited passage. Citation markers are checked against the supplied
  sources.
* **Paper-aware ingestion.** PDFs are split per page and per detected section, so every chunk keeps its page
  number and section heading. Title and year are extracted. Embeddings are batched with bounded retries.
* **Chat-scoped papers.** Each paper belongs to the chat it was uploaded in. A new chat starts with no papers, and
  returning to a chat restores its papers and conversation. Retrieval, comparison and gap analysis only see the
  current chat's papers; the server checks user and chat ownership on every request.
* **Paper selection.** Tick papers in the chat to restrict a question to them.
* **Multi-paper comparison.** Compare 2–6 papers on overview, methods, results or limitations. Every cell is
  cited or marked *not reported in the paper*.
* **Research gaps.** Cross-paper hypotheses (cited evidence from two or more papers) are shown separately
  from single-paper observations.
* **Deep analysis (optional).** A bounded [LangGraph](https://langchain-ai.github.io/langgraph/) workflow
  plans up to 3 searches, checks that every selected paper has evidence, and re-queries at most once.
  It is off by default (`AGENT_ENABLED`).
* **Retrieval modes.** Vector search with pgvector (default). An optional BM25 + vector hybrid sits behind
  `RETRIEVAL_MODE`.
* **Engineering.**
  * Multi-tenant isolation, ownership checks and a prompt-injection guard.
  * Embedding-space isolation per document.
  * Structured logs with request IDs, schema migrations and a `/health` endpoint.
  * API, worker and evaluation test suites in CI.

> Formerly **DocuMind**. The repository and some internal identifiers keep the old name; see
> [Naming](#naming).

## Screenshots

| Cited answer restricted to a selected paper | Side-by-side comparison of two papers |
| --- | --- |
| <img src="docs/screenshots/answer.png" alt="Cited answer" /> | <img src="docs/screenshots/compare.png" alt="Comparison table" /> |

## Architecture

```mermaid
graph LR
    B((Browser)) -->|"Next.js 16 · :3000"| FE[frontend]
    FE -->|"sign up / sign in"| SB
    FE -->|"REST + streamed chat · Bearer token"| API

    subgraph "docker compose (this repo)"
        API["FastAPI :8000<br/>retrieval · prompts · analysis · agent"]
        W["Celery worker<br/>extract · chunk · embed"]
        R[(Redis)]
    end

    subgraph "supabase start (local Supabase)"
        SB["Auth + Storage API :54321"]
        PG[("Postgres + pgvector :54322")]
    end

    O["Ollama on the host :11434<br/>qwen3:8b · nomic-embed-text"]

    API -->|enqueue| R --> W
    API -->|"upload PDF (private bucket)"| SB
    W -->|download PDF| SB
    API <-->|"SQL / vector search"| PG
    W -->|"chunks + vectors"| PG
    API <-->|"chat · JSON analysis · query embedding"| O
    W -->|"chunk embeddings"| O
```

Ollama runs natively, not in Docker, so it can use the GPU (Apple Metal, or CUDA on Linux). Containers reach
it, and the Supabase ports, through `host.docker.internal`.

## Quick start (local, no API keys)

### Prerequisites

| Tool | Version tested | Install |
| --- | --- | --- |
| Docker with Compose v2 (Docker Desktop, OrbStack, or Docker Engine on Linux) | Docker 29, Compose 5 | docs.docker.com/get-docker |
| Supabase CLI | 2.121 | `brew install supabase/tap/supabase`, or use `npx supabase …` (no install) |
| Ollama | 0.40 | `brew install ollama` (macOS) · `curl -fsSL https://ollama.com/install.sh \| sh` (Linux) |
| Node.js + npm | Node 20+ (tested 26), npm 10+ | nodejs.org |

You also need about **12 GB of free disk** (Supabase images ~3 GB, models ~5.5 GB, app images ~1.5 GB) and
**16 GB of RAM** for the default model (see [Local models](#local-models)). Python is only needed to run the
tests.

### 1. Clone and start the services

```bash
git clone https://github.com/AdityaKishore0698/DocuMind_RAG.git researchlens
cd researchlens

# Models (one-time download, ~5.5 GB)
ollama pull qwen3:8b
ollama pull nomic-embed-text
# Keep Ollama running: the desktop app, `brew services start ollama`, or `ollama serve` in another terminal.

# Local Supabase: Postgres + pgvector, auth, storage (first run downloads its images)
supabase start

# Write .env.local (backend) and frontend/.env.local from the running Supabase stack
./scripts/setup_local_env.sh

# API + worker + Redis
docker compose up -d --build
curl -s http://localhost:8000/health        # "ok": true once the database, Redis and Ollama are reachable
```

### 2. Start the frontend

```bash
cd frontend
npm ci
npm run dev                                  # http://localhost:3000
```

### 3. Use it

1. Open <http://localhost:3000>, choose **Register**, and enter any email and a password of at least
   6 characters. The local stack needs no email confirmation, so you are signed in at once.
2. Upload one or more paper PDFs, for example from arXiv:
   `curl -L -o diffusion_policy.pdf https://arxiv.org/pdf/2303.04137` and
   `curl -L -o act.pdf https://arxiv.org/pdf/2304.13705`.
3. Uploading in a new chat creates that chat; the papers belong to it. Wait for **Ready** (seconds per paper).
   Tick papers, then ask a question. Use **Compare N papers**
   for the comparison table and research gaps.

The database schema is created by the API on first start (tables, pgvector extension, indexes). The private
`documents` storage bucket is created by `supabase start` from [`supabase/config.toml`](supabase/config.toml).
Nothing depends on a previously initialised machine.

### Stop, restart, clean up

```bash
docker compose down                 # stop API, worker, Redis (data stays in Supabase)
supabase stop                       # stop Supabase; keeps its data volume
# restart later: supabase start && docker compose up -d   (and `npm run dev` in frontend/)

docker compose down -v --rmi local  # remove containers, the Redis volume and the built images
supabase stop --no-backup           # delete the local database, users and uploaded files
```

## Local models

| Role | Model | Download | Memory when loaded |
| --- | --- | --- | --- |
| Embeddings (ingestion + queries) | `nomic-embed-text` (768-d) | 274 MB | ~0.4 GB |
| Answers, comparisons, research gaps, deep-analysis planning | `qwen3:8b` (Q4_K_M) | 5.2 GB | ~5.6 GB |

* **Hardware.** An Apple Silicon Mac with 16 GB, or a Linux machine with a GPU with at least 8 GB of VRAM,
  runs the defaults comfortably. On CPU only it works, but slowly: minutes per answer.
* **Measured on a MacBook Air M2 (16 GB), local stack:**

  | Step | Time |
  | --- | --- |
  | Ingesting two papers (19 + 18 pages) | ~13 s |
  | Cited answer | 25–45 s |
  | Two-paper comparison | ~70 s |
  | Research gaps for two papers | ~70 s |

  The first call after the model unloads adds ~10 s of loading time; `OLLAMA_KEEP_ALIVE=10m` keeps it loaded.
* **Smaller machines.** Use `ollama pull qwen3:4b` and set `OLLAMA_CHAT_MODEL=qwen3:4b` in `.env.local`
  (untested here; expect weaker comparisons). Changing the chat model does not affect stored vectors.

### Embedding spaces are never mixed

Vectors from different embedding models are not comparable, even when they have the same dimension.
ResearchLens enforces this:

* Every document records the embedding space that produced its vectors, in `documents.embedding_model`
  (e.g. `ollama:nomic-embed-text` or `gemini:gemini-embedding-2`).
* Retrieval only searches documents whose space matches the running configuration. Selecting a paper from
  another space is refused (HTTP 409), and the sidebar marks it **Re-process**.
* **Re-process** re-embeds the stored PDF with the current model, so switching providers never requires
  re-uploading.
* `GET /health` reports the active space and the number of documents in each space. The API logs a warning
  at startup if other spaces exist.
* Local embeddings are refused when `DATABASE_URL` points at a hosted (dotted-hostname) database.

### Papers belong to chats

* **Upload.** `POST /document/upload` requires the chat (`session_id`). Uploading in a new chat creates it first.
* **Scope.** Listing, vector and BM25 retrieval, deep analysis, comparison and gaps all filter on the authenticated
  user **and** the current chat. A paper id from another chat or another user gets the same 404 as a missing one.
* **Re-processing** keeps a paper in its chat and replaces its chunks.
* **Deleting a chat** deletes the chat, its history, its papers and their stored files, after a confirmation that
  says so. Rows are deleted first and files second; a file that cannot be removed is reported in the response and
  shown to the user, never silently ignored.
* **Earlier papers.** Papers uploaded before this existed (migration `004`) are not assigned to any chat
  automatically. They appear under *Earlier papers (not in a chat)* and are searched nowhere until you choose
  **Move here** in a chat. A paper can be moved once; it is never shared between chats.
* **Upgrading a database created by an older version.** Stop the API and run
  `python scripts/migrate.py status --database-url …`, then `up --database-url …` (it applies `003` and `004`;
  no paper, chunk, chat or message is removed). Papers stay unassigned until you move them. Papers with no
  recorded embedding model show **Re-process** after the move; pass `--assume-embedding-model <space>` to `up`
  only if you know which model embedded them.

### Optional: Gemini instead of Ollama

Set `INFERENCE_PROVIDER=gemini` and `GEMINI_API_KEY` in `.env.local` (see [`.env.example`](.env.example)),
then run `docker compose up -d`. Existing papers switch to **Re-process** because Gemini embeddings are a
different space.

### Optional: a remote GPU through an SSH tunnel

If Ollama runs on another machine, bound to its own `127.0.0.1:11434`, forward it to a spare local port and
point the containers at that port:

```bash
# macOS (Docker Desktop / OrbStack reach the host's loopback through host.docker.internal)
ssh -N -L 127.0.0.1:11435:127.0.0.1:11434 <gpu-host>
# Linux: bind the Docker bridge address instead, so containers can reach it without exposing it to the LAN
ssh -N -L 172.17.0.1:11435:127.0.0.1:11434 <gpu-host>
```

Then set `OLLAMA_BASE_URL=http://host.docker.internal:11435` in `.env.local` and run `docker compose up -d`.
The embedding model must be the same (`nomic-embed-text`) for the stored vectors to stay searchable. If the
remote machine serves a different embedding model, the app treats it as a different space, as above.
[`local_inference/README.md`](local_inference/README.md) has more detail.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `/health` → `inference.ok: false`, `ConnectError` | Ollama is not running or not reachable from Docker. Start it (`ollama serve`). On **Linux**, run Ollama with `OLLAMA_HOST=0.0.0.0` so containers can reach it through the host gateway, and keep port 11434 firewalled from the network. |
| `/health` → `missing_models` | Run `ollama pull <model>` for each model listed. |
| `setup_local_env.sh`: "Supabase stack is not running" | Run `supabase start` from the repository root first. |
| `supabase start` fails: port 5432x already allocated | Another local Supabase project is running. Stop it with `supabase stop --project-id <id>` (see `docker ps`). |
| API container restarts with `SchemaMismatch` | The database was created by an older version. Run `python scripts/migrate.py up --database-url postgresql://postgres:postgres@127.0.0.1:54322/postgres`, or start fresh with `supabase stop --no-backup && supabase start`. |
| Sign-in fails with "Invalid login credentials" after `supabase stop --no-backup` | The local users were deleted with the database; register again. |
| Browser shows "Could not reach the API" | Check `docker compose ps`. `FRONTEND_ORIGINS` in `.env.local` must contain the exact origin you browse from (default `http://localhost:3000`). |
| A paper stays in **Processing** | `docker compose logs worker`. Ollama errors are reported in the document's error text; use **Re-process** after fixing. |
| Answers are very slow | The model is running on CPU, or memory is swapping. Close other apps, or use `qwen3:4b`. |
| `npm ci` prints `allow-scripts` warnings | Harmless with npm 11; the app does not need those install scripts. |

## Configuration

[`.env.example`](.env.example) documents every backend variable. The defaults that matter:

| Variable | Default | Meaning |
| --- | --- | --- |
| `INFERENCE_PROVIDER` | `ollama` in `.env.local` | `ollama` or `gemini` (the code default is `gemini` when unset) |
| `OLLAMA_CHAT_MODEL` / `OLLAMA_EMBEDDING_MODEL` | `qwen3:8b` / `nomic-embed-text` | Local models |
| `RETRIEVAL_MODE` | `vector` | `hybrid_bm25` needs `python scripts/migrate.py up 001_bm25_hybrid --database-url …` |
| `RETRIEVAL_TOP_K` | `5` | Chunks per answer |
| `AGENT_ENABLED` | `false` | Shows the **Deep analysis** toggle in the chat |
| `MAX_UPLOAD_MB` | `20` | Per-file upload limit (PDF/TXT only) |

## Development

```bash
# Tests (Python 3.11; a throwaway pgvector database on port 5435; all model calls are faked)
docker run -d --rm --name researchlens-test-db -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=researchlens_test -p 127.0.0.1:5435:5432 pgvector/pgvector:pg16
python3.11 -m venv .venv          # or: uv venv --python 3.11 .venv
.venv/bin/pip install -r api/requirements.txt -r worker/requirements.txt   # or: uv pip install --python .venv/bin/python -r …
export TEST_DB=postgresql://postgres:postgres@127.0.0.1:5435/researchlens_test
(cd api && DATABASE_URL=$TEST_DB TEST_DATABASE_URL=$TEST_DB ../.venv/bin/python -m pytest -q)
(cd worker && WORKER_TEST_DATABASE_URL=$TEST_DB ../.venv/bin/python -m pytest -q tests)
(cd frontend && npm run lint && npm run build)
docker stop researchlens-test-db
```

* **Schema changes.** Use `python scripts/migrate.py status|up|down --database-url <loopback URL>` (provenance
  `002`, embedding space `003`, chat-scoped papers `004`, optional BM25 `001`). It refuses non-local databases.
* **Evaluation.** The retrieval and answer benchmark harness is in [`evaluation/`](evaluation/README.md).
  Results and design records are in [`docs/`](docs/): [architecture](docs/architecture.md),
  [technical decisions](docs/technical-decisions.md), [evaluation](docs/evaluation.md),
  [failure analysis](docs/failure-analysis.md).

## Naming

User-facing material uses **ResearchLens — Research Paper Analyst**. The following keep the old name
deliberately, because renaming them would break existing databases, saved browser settings, or evaluation
reproducibility:

* the GitHub repository (`DocuMind_RAG`);
* the logger namespace `documind.*`;
* the database objects `documind_migrations`, `documind_chunks_fts` and `documind_token_count`;
* the browser storage keys `documind-*`;
* the evaluation database names and the frozen evaluation prompts.

Legacy deployment files from the hosted version are not used by the local setup: `docker-compose.prod.yml`,
`render.yaml`, the root `Dockerfile`, `start.sh` and the Streamlit prototype in `ui/`.
