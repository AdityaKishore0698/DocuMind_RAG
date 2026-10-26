# Local inference (Ollama)

ResearchLens's default setup (see the main [README](../README.md#quick-start-local-no-api-keys)) runs all model calls
through a local [Ollama](https://ollama.com) server:

| Role | Variable | Default |
| --- | --- | --- |
| Document and query embeddings (768-d) | `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` |
| Answers, comparisons, research gaps, deep-analysis planning/assessment (JSON) | `OLLAMA_CHAT_MODEL` | `qwen3:8b` |

The provider boundary is `api/core/inference.py` (chat, structured JSON, query embeddings) and
`worker/inference.py` (batched document embeddings). There is no silent fallback: if Ollama is unreachable the
request fails with a clear error (`/health` shows why).

This folder holds an optional diagnostic tool and templates for running it outside Docker.

## Hardware and measured performance

Measured on a MacBook Air M2 with 16 GB unified memory, Ollama 0.40 (Metal):

| Step | Result |
| --- | --- |
| Memory with both models loaded | ~5.6 GB (`qwen3:8b`) + ~0.4 GB (`nomic-embed-text`) |
| First generation after a cold start | ~11 s to the first token (model load) |
| Ingest two papers (37 pages, 227 chunks) | ~13 s |
| Cited answer | 25–45 s |
| Two-paper comparison / gap analysis | ~70 s each |

* **16 GB of RAM or VRAM** is comfortable for `qwen3:8b`.
* With 8 GB, try `qwen3:4b` (untested here).
* `qwen3:14b` needs ~11 GB of VRAM. It was downloaded on a shared lab GPU server, but inference with it was
  **not tested**: the GPU lacked headroom, and later SSH access to the server timed out.
* CPU-only machines work but take minutes per answer.

## Embedding spaces

Changing `OLLAMA_EMBEDDING_MODEL`, toggling `OLLAMA_EMBEDDING_PREFIXES`, or switching to Gemini changes the
embedding space:

* Each document records its space (`documents.embedding_model`), and retrieval only searches documents in the
  active space.
* Other papers show **Re-process** in the sidebar until they are re-embedded.
* Changing only the **chat** model has no effect on stored vectors.
* Local embeddings are refused when `DATABASE_URL` points at a hosted database.

## Using Ollama on another machine (SSH tunnel)

If a GPU server runs Ollama on its own `127.0.0.1:11434`, forward it to a local port, then point the app at that
port.

1. Start the tunnel and check it:

   ```bash
   # macOS: Docker Desktop / OrbStack reach host loopback via host.docker.internal
   ssh -N -L 127.0.0.1:11435:127.0.0.1:11434 <gpu-host>
   # Linux: bind the Docker bridge address so containers can reach it (not the whole LAN)
   ssh -N -L 172.17.0.1:11435:127.0.0.1:11434 <gpu-host>

   curl -s http://127.0.0.1:11435/api/tags   # lists the remote models (use 172.17.0.1 on Linux)
   ```

2. In `.env.local`, set `OLLAMA_BASE_URL=http://host.docker.internal:11435`, then run `docker compose up -d`.
3. Keep `OLLAMA_EMBEDDING_MODEL` the same as before (e.g. `nomic-embed-text` on both machines) to keep existing
   papers searchable. A different embedding model on the server is a different space, and papers must be
   re-processed.

**Shared GPUs.** Check headroom before loading a model:

```bash
ssh <gpu-host> 'nvidia-smi --query-gpu=memory.free,utilization.gpu,temperature.gpu --format=csv'
```

`qwen3:14b` at `num_ctx` 8192 needs ~11 GB free. On a busy GPU, Ollama offloads layers to the CPU and competes
with other users' jobs. `OLLAMA_KEEP_ALIVE=2m` frees VRAM soon after use.

## Diagnostics without Docker: `check.py`

`check.py` runs staged checks against the application's own provider code, cheapest first:

```bash
python3.11 -m venv local_inference/.venv
local_inference/.venv/bin/pip install -r local_inference/requirements.txt
cp local_inference/env.mac.example local_inference/.env.mac          # git-ignored; edit if needed
set -a; source local_inference/.env.mac; set +a
local_inference/.venv/bin/python local_inference/check.py            # server version + model metadata (loads nothing)
local_inference/.venv/bin/python local_inference/check.py --embed    # one embedding
local_inference/.venv/bin/python local_inference/check.py --generate # one short generation
local_inference/.venv/bin/python local_inference/check.py --rag      # synthetic ingest → retrieve → answer
```

`--rag` needs an isolated pgvector database:

```bash
docker run -d --name researchlens-check-db -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=documind_local_ollama \
  -p 127.0.0.1:5436:5432 pgvector/pgvector:pg16
```

`env.example` is the same template for a remote GPU reached through a tunnel on port 11435.

## Downloading models behind a restrictive proxy

`ollama pull` opens direct connections to the model CDN. On networks that only allow traffic through an HTTP
proxy, [`fetch_model_via_proxy.sh`](fetch_model_via_proxy.sh) downloads the registry blobs with `curl` through
the proxy, checks each sha256, and writes them into `~/.ollama/models` (for example
`./fetch_model_via_proxy.sh qwen3 8b`).

## Common errors

| Error | Fix |
| --- | --- |
| `Ollama embedding failed … ConnectError` | Ollama is not running or not reachable at `OLLAMA_BASE_URL`. From containers, use `host.docker.internal`; on Linux, also run Ollama with `OLLAMA_HOST=0.0.0.0`. |
| `model "…" not found` | Run `ollama pull <model>`. `/health` lists the missing models. |
| `OLLAMA_CHAT_MODEL is required` | Set it (`qwen3:8b`); there is no default, so a machine never loads an unintended model. |
| `requires a local DATABASE_URL` | Local embeddings were pointed at a hosted database; use the local Supabase database. |
| Answers time out | The first call loads the model; raise `OLLAMA_READ_TIMEOUT_S`, or check memory pressure. |
