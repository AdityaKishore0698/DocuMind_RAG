"""Embedding provider for ingestion: Gemini (default) or a local Ollama server.

Embedding-only counterpart of api/core/inference.py (the worker is a separate image).

* `get_embeddings(texts)` embeds in batches (plan §7.1): Gemini gets one `types.Content` per text — a plain
  `list[str]` would be merged into ONE embedding by the SDK — and the returned count is checked; Ollama gets
  a list `input`. Requests are paced by embedded *items* per minute (the Gemini free tier counts items),
  429/5xx/timeouts are retried a bounded number of times honouring the server's `retryDelay`, and a daily
  quota raises `EmbeddingQuotaExhausted` at once (retrying cannot succeed for hours).
* `INFERENCE_PROVIDER=ollama` uses `OLLAMA_EMBEDDING_MODEL` (default `nomic-embed-text`, document prefix
  `search_document: `), never falls back to Gemini, and refuses a non-local DATABASE_URL — nomic vectors are a
  different embedding space from the Gemini vectors in the shared database.
* Errors carry the provider's status/category only — never chunk text.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from urllib.parse import urlparse

EMBEDDING_DIM = 768
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
MAX_WAIT_S = 60.0
LONG_RETRY_S = 600.0


class InferenceConfigError(RuntimeError):
    pass


class InferenceUnavailable(RuntimeError):
    pass


class EmbeddingQuotaExhausted(InferenceUnavailable):
    """A daily (long-window) quota is exhausted; retry later rather than now."""


def provider_name() -> str:
    name = os.getenv("INFERENCE_PROVIDER", "gemini").strip().lower()
    if name not in ("gemini", "ollama"):
        raise InferenceConfigError(f"INFERENCE_PROVIDER must be 'gemini' or 'ollama', got {name!r}")
    return name


def is_local_host(host: str | None) -> bool:
    """Same rule as api/core/inference.py: loopback, Docker's host alias, or a single-label service name."""
    return bool(host) and (host in LOOPBACK_HOSTS or host == "host.docker.internal" or "." not in host
                           and ":" not in host)


def embedding_space() -> str:
    """Identity of the embedding space this worker writes; must equal api/core/inference.py `embedding_space()`."""
    if provider_name() == "ollama":
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text").strip()
        no_prefix = os.getenv("OLLAMA_EMBEDDING_PREFIXES", "true").strip().lower() in {"0", "false", "no", "off"}
        return f"ollama:{model}" + ("+noprefix" if no_prefix else "")
    return "gemini:" + os.getenv("EMBEDDING_MODEL", "gemini-embedding-2").strip()


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


# --- pacing -------------------------------------------------------------------------------------------
class ItemPacer:
    """Sliding 60-second budget of embedded items (0 disables)."""

    def __init__(self, items_per_minute: int, clock=time.monotonic, sleep=time.sleep):
        self.limit, self.clock, self.sleep = items_per_minute, clock, sleep
        self.events: deque[tuple[float, int]] = deque()

    def acquire(self, n: int) -> None:
        if not self.limit:
            return
        n = min(n, self.limit)
        while True:
            now = self.clock()
            while self.events and self.events[0][0] <= now - 60:
                self.events.popleft()
            if sum(k for _, k in self.events) + n <= self.limit:
                self.events.append((now, n))
                return
            self.sleep(max(0.05, self.events[0][0] + 60 - now))


# --- Gemini error inspection ------------------------------------------------------------------------------
def _payload(exc) -> dict:
    details = getattr(exc, "details", None)
    if isinstance(details, list) and len(details) == 1:
        details = details[0]
    return details.get("error", details) if isinstance(details, dict) else {}


def _duration(text: str) -> float | None:
    m = re.match(r"^\s*(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:(\d+(?:\.\d+)?)s)?\s*$", text or "")
    if not m or not any(m.groups()):
        return None
    h, mins, s = m.groups()
    return int(h or 0) * 3600 + int(mins or 0) * 60 + float(s or 0)


def retry_after(exc) -> float | None:
    payload = _payload(exc)
    for item in payload.get("details", []) or []:
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
            value = _duration(str(item.get("retryDelay", "")))
            if value is not None:
                return value
    m = re.search(r"retry in\s+((?:\d+h)?\s*(?:\d+m(?!s))?\s*(?:\d+(?:\.\d+)?s)?)", str(payload.get("message", "")), re.I)
    return _duration(m.group(1)) if m else None


def classify(exc) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        if code == 429:
            quota_ids = [str(v.get("quotaId", "")) for item in _payload(exc).get("details", []) or []
                         if isinstance(item, dict) for v in item.get("violations", []) or []]
            delay = retry_after(exc)
            if any("perday" in q.lower() for q in quota_ids) or (delay is not None and delay > LONG_RETRY_S):
                return "daily_quota"
            return "rate_limit"
        return "server" if code >= 500 else f"client_{code}"
    return "timeout" if "timeout" in type(exc).__name__.lower() else "other"


# --- providers -------------------------------------------------------------------------------------------------
_genai_client = None


def _gemini_client():
    global _genai_client
    from google import genai

    if _genai_client is None:
        _genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _genai_client


def _gemini_batch(texts: list[str]) -> list[list[float]]:
    from google.genai import types

    contents = [types.Content(role="user", parts=[types.Part(text=t)]) for t in texts]
    result = _gemini_client().models.embed_content(
        model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-2"), contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM))
    vectors = [list(e.values) for e in (result.embeddings or [])]
    if len(vectors) != len(texts):
        raise InferenceUnavailable(f"Gemini returned {len(vectors)} embeddings for {len(texts)} inputs")
    return vectors


def _ollama_batch(texts: list[str], transport=None) -> list[list[float]]:
    import httpx

    if not is_local_host(urlparse(os.getenv("DATABASE_URL") or "").hostname):
        raise InferenceConfigError("INFERENCE_PROVIDER=ollama requires a local DATABASE_URL (localhost, "
                                   "host.docker.internal or a Compose service) — local embeddings must never "
                                   "enter a hosted database")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    prefix = "" if os.getenv("OLLAMA_EMBEDDING_PREFIXES", "true").lower() in {"0", "false", "no", "off"} \
        else "search_document: "
    timeout = httpx.Timeout(float(os.getenv("OLLAMA_READ_TIMEOUT_S", "300")),
                            connect=float(os.getenv("OLLAMA_CONNECT_TIMEOUT_S", "5")))
    payload = {"model": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
               "input": [prefix + t for t in texts], "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "2m")}
    try:
        # trust_env=False: the endpoint is a private tunnel; never route it via a system HTTP proxy.
        with httpx.Client(timeout=timeout, transport=transport, trust_env=False) as client:
            response = client.post(f"{base_url}/api/embed", json=payload)
    except httpx.HTTPError as exc:  # endpoint + error class only; never the chunk text
        raise InferenceUnavailable(f"Ollama embedding failed at {base_url}: {type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        raise InferenceUnavailable(f"Ollama embedding returned HTTP {response.status_code}")
    vectors = response.json().get("embeddings") or []
    if len(vectors) != len(texts) or any(len(v) != EMBEDDING_DIM for v in vectors):
        raise InferenceConfigError(f"unexpected embedding shape from Ollama (expected {len(texts)} x {EMBEDDING_DIM})")
    return [[float(x) for x in v] for v in vectors]


def get_embeddings(texts: list[str], *, batch_fn=None, pacer: ItemPacer | None = None, sleep=time.sleep,
                   max_retries: int | None = None) -> list[list[float]]:
    """Embed `texts` in order, batched, paced and with bounded retries."""
    ollama = provider_name() == "ollama"
    batch_fn = batch_fn or (_ollama_batch if ollama else _gemini_batch)
    size = max(1, _int_env("EMBED_BATCH_SIZE", 32))
    pacer = pacer or ItemPacer(0 if ollama else _int_env("EMBED_ITEMS_PER_MIN", 90), sleep=sleep)
    retries = _int_env("EMBED_MAX_RETRIES", 4) if max_retries is None else max_retries
    vectors: list[list[float]] = []
    for start in range(0, len(texts), size):
        batch = texts[start:start + size]
        attempt = 0
        while True:
            pacer.acquire(len(batch))
            try:
                vectors.extend(batch_fn(batch))
                break
            except (InferenceConfigError, InferenceUnavailable):
                raise
            except Exception as exc:  # noqa: BLE001 — provider SDK errors are classified below
                category = classify(exc)
                if category == "daily_quota":
                    raise EmbeddingQuotaExhausted("embedding quota exhausted for today — retry later") from exc
                if category not in ("rate_limit", "server", "timeout") or attempt >= retries:
                    raise InferenceUnavailable(f"embedding failed ({category}) after {attempt + 1} attempt(s)") from exc
                attempt += 1
                server = retry_after(exc)
                sleep(min(MAX_WAIT_S, max(2.0 ** attempt, (server or 0) + 2.0)))
    return vectors


def get_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]
