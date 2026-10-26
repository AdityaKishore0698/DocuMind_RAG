"""Ollama backend for the local-model baseline (System L).

Drop-in counterparts of ``gemini.Embedder`` and ``gemini.stream_generate`` so the
rest of the harness (ingest, run, resume, report) is shared. Self-contained on
purpose — like baseline_a.py, the benchmark must not change when the application's
own provider code (api/core/inference.py) evolves.

* Embeddings: ``/api/embed`` with ``nomic-embed-text``; nomic is trained with task
  prefixes, so documents are embedded as ``search_document: <chunk>`` and queries as
  ``search_query: <question>`` (the same convention as the app's Ollama provider).
  The cache key is the prefixed text, so query and document vectors never collide.
* Generation: ``/api/chat`` streaming NDJSON, ``think=false``, ``num_ctx`` 8192,
  otherwise the model's default sampling (recorded from ``/api/show``).
* Bounded retries only for connection errors, timeouts and 5xx; no quotas apply.
* httpx ``trust_env=False``: the endpoint is local; a system proxy must not intercept it.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import httpx

from gemini import CallStats, sanitize_message

DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "
OPTIONS = {"num_ctx": 8192}
THINK = False
KEEP_ALIVE = "2m"


def base_url() -> str:
    return os.environ.get("EVAL_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(float(os.environ.get("EVAL_OLLAMA_READ_TIMEOUT_S", "600")), connect=5.0)


def _category(exc: BaseException | None, status: int | None = None) -> str:
    if status is not None:
        return "server" if status >= 500 else f"client_{status}"
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.ConnectError):
        return "unavailable"
    return "other"


def _retryable(category: str) -> bool:
    return category in {"server", "timeout", "unavailable"}


def _error(category: str, exc: BaseException | None = None, status: int | None = None, message: str = "") -> dict:
    return {"category": category, "type": type(exc).__name__ if exc else "HTTPStatus", "http_status": status,
            "status": None, "message": sanitize_message(message or (str(exc) if exc else ""))}


class OllamaError(RuntimeError):
    def __init__(self, kind: str, details: dict):
        super().__init__(f"{kind}: {details['category']} — {details['message']}")
        self.kind, self.details = kind, details


class OllamaEmbedder:
    """Same interface as gemini.Embedder: embed_many(texts) for chunks, embed_query(text)."""

    def __init__(self, *, model: str, dim: int, stats: CallStats, conn=None, batch_size: int = 32,
                 max_retries: int = 2, transport=None, sleep=time.sleep):
        self.model, self.dim, self.stats, self.conn = model, dim, stats, conn
        self.batch_size, self.max_retries, self.transport, self.sleep = batch_size, max_retries, transport, sleep
        self.pacer = type("NoPacer", (), {"limit": 0})()  # interface parity for log lines
        self.mode = "batch"

    def _call(self, kind: str, inputs: list[str]) -> tuple[list[list[float]], float]:
        attempt = 0
        while True:
            self.stats.bump(self.stats.attempts, kind)
            self.stats.embed_items_attempted += len(inputs)
            started = time.perf_counter()
            details = None
            try:
                with httpx.Client(timeout=_timeout(), transport=self.transport, trust_env=False) as client:
                    r = client.post(f"{base_url()}/api/embed",
                                    json={"model": self.model, "input": inputs, "keep_alive": KEEP_ALIVE})
                if r.status_code == 200:
                    vectors = r.json().get("embeddings") or []
                    if len(vectors) != len(inputs) or any(len(v) != self.dim for v in vectors):
                        raise RuntimeError(f"Ollama returned {len(vectors)} vectors for {len(inputs)} inputs "
                                           f"(expected {self.dim} dims each)")
                    self.stats.bump(self.stats.requests, kind)
                    self.stats.embed_items_embedded += len(inputs)
                    self.stats.embed_chars_sent += sum(len(t) for t in inputs)
                    return [[float(x) for x in v] for v in vectors], time.perf_counter() - started
                details = _error(_category(None, r.status_code), status=r.status_code, message=r.text[:300])
            except httpx.HTTPError as exc:
                details = _error(_category(exc), exc)
            if attempt == 0:
                self.stats.bump(self.stats.first_attempt_errors, f"{kind}:{details['category']}")
            self.stats.errors.append({"kind": kind, "attempt": attempt + 1, **details,
                                      "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            if not _retryable(details["category"]) or attempt >= self.max_retries:
                self.stats.bump(self.stats.final_failures, f"{kind}:{details['category']}")
                raise OllamaError(kind, details)
            attempt += 1
            self.stats.retries += 1
            wait = min(30.0, 2.0 ** attempt)
            self.stats.retry_sleep_s += wait
            self.sleep(wait)

    def _cached(self, texts: list[str]) -> dict[str, list[float]]:
        from db import cached_embeddings

        return cached_embeddings(self.conn, self.model, self.dim, texts) if self.conn is not None else {}

    def _store(self, pairs) -> None:
        from db import store_embeddings

        if self.conn is not None and pairs:
            store_embeddings(self.conn, self.model, self.dim, pairs)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        from db import text_sha256

        prefixed = [DOC_PREFIX + t for t in texts]
        self.stats.embed_items_requested += len(prefixed)
        cache = self._cached(prefixed)
        missing = list(dict.fromkeys(t for t in prefixed if text_sha256(t) not in cache))
        self.stats.embed_items_from_cache += len(prefixed) - len(missing)
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            vectors, _ = self._call("embed_batch", batch)
            pairs = list(zip(batch, vectors))
            self._store(pairs)  # cached per batch: resumable
            cache.update({text_sha256(t): v for t, v in pairs})
        return [cache[text_sha256(t)] for t in prefixed]

    def embed_query(self, text: str) -> tuple[list[float], float, bool]:
        from db import text_sha256

        prefixed = QUERY_PREFIX + text
        hit = self._cached([prefixed]).get(text_sha256(prefixed))
        if hit is not None:
            return hit, 0.0, True
        vectors, seconds = self._call("embed_query", [prefixed])
        self._store([(prefixed, vectors[0])])
        return vectors[0], seconds, False


async def stream_generate(*, model: str, system_instruction: str, question: str, stats: CallStats,
                          max_retries: int = 2, transport=None) -> dict:
    """Same result shape as gemini.stream_generate. A failure after text has streamed is
    returned as a partial answer (never retried, never a success)."""
    payload = {"model": model, "stream": True, "think": THINK, "keep_alive": KEEP_ALIVE, "options": OPTIONS,
               "messages": [{"role": "system", "content": system_instruction},
                            {"role": "user", "content": question}]}
    attempt = 0
    while True:
        stats.bump(stats.attempts, "generate")
        started = time.perf_counter()
        text, ttft, final, details = "", None, {}, None
        try:
            async with httpx.AsyncClient(timeout=_timeout(), transport=transport, trust_env=False) as client:
                async with client.stream("POST", f"{base_url()}/api/chat", json=payload) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode(errors="replace")
                        details = _error(_category(None, response.status_code), status=response.status_code,
                                         message=body[:300])
                    else:
                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue
                            event = json.loads(line)
                            if event.get("error"):
                                details = _error("server", message=str(event["error"]))
                                break
                            piece = (event.get("message") or {}).get("content") or ""
                            if piece:
                                ttft = ttft if ttft is not None else time.perf_counter() - started
                                text += piece
                            if event.get("done"):
                                final = event
                                break
                        if details is None and not final:
                            details = _error("server", message="stream ended without done=true")
        except httpx.HTTPError as exc:
            details = _error(_category(exc), exc)
        total = time.perf_counter() - started
        if details is None:
            stats.bump(stats.requests, "generate")
            usage = {"prompt_token_count": final.get("prompt_eval_count"),
                     "candidates_token_count": final.get("eval_count")}
            usage = {k: int(v) for k, v in usage.items() if v is not None}
            if usage:
                usage["total_token_count"] = sum(usage.values())
            for k, v in usage.items():
                stats.bump(stats.tokens, k, v)
            timings = {k: round(final[k] / 1e9, 3) for k in ("load_duration", "prompt_eval_duration", "eval_duration")
                       if final.get(k) is not None}
            return {"text": text, "ttft_s": ttft, "total_s": total, "usage": usage, "attempts": attempt + 1,
                    "error": None, "server_timings_s": timings}
        if attempt == 0:
            stats.bump(stats.first_attempt_errors, f"generate:{details['category']}")
        stats.errors.append({"kind": "generate", "attempt": attempt + 1, **details,
                             "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if text or not _retryable(details["category"]) or attempt >= max_retries:
            stats.bump(stats.final_failures, f"generate:{details['category']}")
            return {"text": text, "ttft_s": ttft, "total_s": total, "usage": {}, "attempts": attempt + 1,
                    "error": {**details, "retry_budget_exhausted": _retryable(details["category"]) and not text,
                              "daily_quota": False, "partial_answer": bool(text), "quota": None,
                              "retry_after_s": None}}
        attempt += 1
        stats.retries += 1
        wait = min(30.0, 2.0 ** attempt)
        stats.retry_sleep_s += wait
        await asyncio.sleep(wait)


def model_metadata(models: list[str], transport=None) -> dict:
    """Digest, size, quantization and default sampling parameters of each model."""
    out = {}
    with httpx.Client(timeout=httpx.Timeout(15, connect=5), transport=transport, trust_env=False) as client:
        tags = {m["name"]: m for m in client.get(f"{base_url()}/api/tags").json().get("models", [])}
        version = client.get(f"{base_url()}/api/version").json().get("version")
        for name in models:
            show = client.post(f"{base_url()}/api/show", json={"model": name})
            if show.status_code != 200:
                out[name] = {"error": show.text[:200]}
                continue
            info = show.json()
            tag = tags.get(name) or tags.get(f"{name}:latest") or {}
            out[name] = {"digest": tag.get("digest"), "size": tag.get("size"),
                         "parameter_size": info.get("details", {}).get("parameter_size"),
                         "quantization": info.get("details", {}).get("quantization_level"),
                         "default_parameters": info.get("parameters")}
    return {"ollama_version": version, "base_url": base_url(), "models": out,
            "request_options": {"think": THINK, "keep_alive": KEEP_ALIVE, **OPTIONS},
            "embedding_prefixes": {"document": DOC_PREFIX, "query": QUERY_PREFIX}}


def chat_json(*, model: str, messages: list[dict], schema: dict, stats: CallStats, kind: str,
              max_retries: int = 2, transport=None, sleep=time.sleep) -> tuple[dict | None, dict]:
    """One non-streaming structured call (Ollama ``format`` = JSON schema). Returns (parsed or None, info).
    A response that is not valid JSON is returned as None (callers fall back deterministically) and
    counted in ``info["parse_error"]``; transport failures raise OllamaError after bounded retries."""
    payload = {"model": model, "messages": messages, "stream": False, "think": THINK, "keep_alive": KEEP_ALIVE,
               "format": schema, "options": {**OPTIONS, "temperature": 0}}
    attempt = 0
    while True:
        stats.bump(stats.attempts, kind)
        started = time.perf_counter()
        details = None
        try:
            with httpx.Client(timeout=_timeout(), transport=transport, trust_env=False) as client:
                r = client.post(f"{base_url()}/api/chat", json=payload)
            if r.status_code == 200:
                body = r.json()
                stats.bump(stats.requests, kind)
                usage = {"prompt_token_count": body.get("prompt_eval_count"), "candidates_token_count": body.get("eval_count")}
                usage = {k: int(v) for k, v in usage.items() if v is not None}
                if usage:
                    usage["total_token_count"] = sum(usage.values())
                for k, v in usage.items():
                    stats.bump(stats.tokens, k, v)
                info = {"seconds": time.perf_counter() - started, "usage": usage, "parse_error": False}
                try:
                    parsed = json.loads((body.get("message") or {}).get("content") or "")
                    return (parsed if isinstance(parsed, dict) else None), {**info, "parse_error": not isinstance(parsed, dict)}
                except json.JSONDecodeError:
                    return None, {**info, "parse_error": True}
            details = _error(_category(None, r.status_code), status=r.status_code, message=r.text[:300])
        except httpx.HTTPError as exc:
            details = _error(_category(exc), exc)
        if attempt == 0:
            stats.bump(stats.first_attempt_errors, f"{kind}:{details['category']}")
        stats.errors.append({"kind": kind, "attempt": attempt + 1, **details,
                             "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        if not _retryable(details["category"]) or attempt >= max_retries:
            stats.bump(stats.final_failures, f"{kind}:{details['category']}")
            raise OllamaError(kind, details)
        attempt += 1
        stats.retries += 1
        wait = min(30.0, 2.0 ** attempt)
        stats.retry_sleep_s += wait
        sleep(wait)
