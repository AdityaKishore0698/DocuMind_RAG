# Local inference (Ollama) — experimental

Gemini remains the default; nothing changes unless `INFERENCE_PROVIDER=ollama`. Same code, two
environments, selected only by environment files (no model names hard-coded in the app):

| | Lab server (`<lab-host>`) | Mac (development) |
|---|---|---|
| Branch | `feature/local-inference-qwen3-14b` | `feature/local-inference-qwen3-8b-mac` |
| Template | `env.example` → `.env.lab` | `env.mac.example` → `.env.mac` |
| Generation | `qwen3:14b` (Q4_K_M, installed) | `qwen3:8b` (Q4_K_M, installed, verified) |
| Embeddings | `nomic-embed-text` | `nomic-embed-text` |
| Endpoint | `127.0.0.1:11435` → SSH tunnel → <lab-host>'s system Ollama | `127.0.0.1:11434`, the Mac's own Ollama |
| Database | isolated local pgvector, port 5436 | same |

`OLLAMA_CHAT_MODEL` is required (no default) so a machine can't silently load the wrong model.
Both `.env.lab` and `.env.mac` are git-ignored (`.env.*`).

## Mac status (verified 2026-10-15, MacBook Air M2, 16 GB unified memory)
* **Ollama 0.40.4** via Homebrew (`brew install ollama`, pulled the `mlx` dependency), run as a user
  launchd service: `brew services start|stop ollama`; listens on 127.0.0.1:11434 only. The service sets
  `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` (Homebrew defaults; the lab server's do not).
  The old dangling `/usr/local/bin/ollama` symlink (root-owned) was left alone; `/opt/homebrew/bin` wins in PATH.
* **Models** (`ollama list`): `qwen3:8b` `500a1f067a9f` 5.2 GB (8.2B, Q4_K_M, 36 layers, 8 KV heads, ctx 40,960);
  `nomic-embed-text` `0a109f422b47` 274 MB (F16, 768-d, ctx 2,048); pre-existing `llama3` kept.
* **Download note:** `ollama pull` cannot work on this network (direct HTTPS is blocked; Ollama 0.40.4
  opens direct connections to the blob CDN even with `HTTPS_PROXY` set). The models were fetched from
  `registry.ollama.ai` with curl through the proxy, every blob sha256-verified, and written into
  `~/.ollama/models` in the registry layout (manifest last): `local_inference/fetch_model_via_proxy.sh qwen3 8b`.
  See docs/failure-analysis.md F-12.
* **Checks** (`.env.mac`): metadata ✅ · `--embed` ✅ · `--generate` ✅ · `--rag` ✅ — numbers below.

| Stage | Result |
|---|---|
| `--embed` (cold, incl. model load) | 768 dims, all finite, L2 norm 1.0, 0.97 s; nomic resident 0.38 GB |
| `--generate` (cold, incl. loading qwen3:8b) | TTFT 11.43 s, total 13.04 s, 106 chars, coherent; qwen3:8b resident 5.63 GB |
| `--rag` (models warm) | correct doc ranked 1st (cosine distance 0.100 vs 0.328), grounded answer, TTFT 1.07 s, total 2.90 s |
| Memory | free 58 % → 20 % with both models loaded; macOS pressure level 2 (warn, not critical); swap 0.37 → 1.54 GB; after unload level 1, 70 % free |

Single runs on synthetic data — smoke tests, not benchmarks.

## Environment (on the Mac, where the app and tests run)
```bash
uv venv --python 3.11 local_inference/.venv
uv pip sync --python local_inference/.venv/bin/python local_inference/requirements.txt
```
`requirements.txt` is the uv-compiled lock of api + worker deps (+ `sqlalchemy<2.1`, see
docs/failure-analysis.md F-9). No torch/CUDA: the Mac only speaks HTTP to Ollama.

## Tunnel (only this task's own process)
```bash
SOCK="$HOME/.ssh/documind-ollama-tunnel.sock"
# start: local 127.0.0.1:11435 -> lab 127.0.0.1:11434 (Ollama stays bound to loopback on the server)
ssh -o ExitOnForwardFailure=yes -M -S "$SOCK" -f -N -L 127.0.0.1:11435:127.0.0.1:11434 <lab-host>
ssh -S "$SOCK" -O check <lab-host>     # status
ssh -S "$SOCK" -O exit  <lab-host>     # stop — closes only this control socket's tunnel
```
Port 11435 avoids clashing with any Ollama on the Mac. The control socket identifies the tunnel
unambiguously, so stopping it never touches other SSH sessions.

## Configuration
Copy the template for the machine (`env.example` → `.env.lab`, `env.mac.example` → `.env.mac`) and
export it. `INFERENCE_PROVIDER=ollama` requires `DATABASE_URL` on localhost (embedding-space guard).

## Checks (cheapest first)
```bash
set -a; source local_inference/.env.lab; set +a     # or .env.mac on the Mac
PY=local_inference/.venv/bin/python
$PY local_inference/check.py              # version + model metadata — loads nothing
$PY local_inference/check.py --embed      # loads nomic-embed-text (~0.3 GB VRAM)
$PY local_inference/check.py --generate   # loads qwen3:14b (Q4_K_M ≈ 9.3 GB weights + KV cache)
$PY local_inference/check.py --rag        # synthetic 3-doc ingest/retrieve/answer in LOCAL_INFERENCE_DATABASE_URL
```
Before any inference stage, check the shared GPU (read-only):
```bash
ssh <lab-host> 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv'
```
Proceed only with enough free VRAM for the model (≈ 11 GB for qwen3:14b at num_ctx 8192) and a
GPU that is not near its thermal limit. Otherwise Ollama offloads layers to CPU and competes with
other users' jobs.

Isolated database for `--rag` (separate from the Gemini evaluation DB on 5434):
```bash
docker run -d --name documind-local-ollama-db -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=documind_local_ollama -p 127.0.0.1:5436:5432 pgvector/pgvector:pg16
```

## Model download (approved, done 2026-10-15)
`ollama pull qwen3:14b` with the existing system Ollama (0.20.3, unchanged service/config/store path):
digest `bdbd181c33f2…`, 9,276,198,565 bytes, sha256 verified by Ollama. Disk 106 → 97 GB free.
Verified metadata: 14.77B params, Q4_K_M, 40 layers, 40 query / 8 KV heads × 128, context 40,960,
capabilities completion/tools/thinking. `nomic-embed-text`: 137M, F16, 768-d, context 2,048.

## Blocked step
Inference stages (`--embed`, `--generate`, `--rag`) — deferred while the GPU is saturated
(14:37 IST: 19.0/24.6 GB used, 100 % utilization, 86 °C). Run them when ≥ 12 GB VRAM is free,
utilization ≤ 80 % and temperature ≤ 80 °C.
