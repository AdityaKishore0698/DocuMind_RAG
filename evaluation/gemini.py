"""Gemini calls for the harness, with usage accounting.

* Every attempt is counted (by kind), timed (excluding pacing/back-off sleeps)
  and classified on failure. ``first_attempt_errors`` records what production
  would have experienced (production has no retries).
* Embeddings are paced by **items** (texts embedded) per minute, not just by
  requests: the Gemini free tier limits ``gemini-embedding-2`` by embedded
  contents per minute, so one batch of 50 texts uses 50 units of quota.
* On 429/5xx/timeouts the harness waits at least the server's ``retryDelay``
  (plus a buffer), using bounded exponential back-off with jitter, and gives up
  after ``max_retries`` — it never retries indefinitely. A 429 for a *daily*
  quota (per-day quotaId, or a delay such as ``16h30m39s``) is never retried. The SDK's own retry is
  off by default (google-genai stops after one attempt), so the harness is the
  only retrier and the counts below are exact.
* Embeddings are cached in the local DB, keyed by (model, dim, sha256(text)).

Batching note — for ``gemini-embedding-2`` the SDK merges a plain ``list[str]``
into ONE multi-part Content (one aggregated embedding). The harness sends one
``types.Content`` per chunk and verifies the returned count, falling back to
per-item calls on any mismatch.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger("evaluation.gemini")

RETRY_BUFFER_S = 2.0       # added on top of the server's requested delay
MAX_SINGLE_WAIT_S = 120.0  # cap for any one back-off sleep
MAX_ERROR_LOG = 50         # bounded error log kept in stats


@dataclass
class CallStats:
    requests: dict[str, int] = field(default_factory=dict)              # kind -> successful requests
    attempts: dict[str, int] = field(default_factory=dict)              # kind -> total attempts (incl. failed)
    first_attempt_errors: dict[str, int] = field(default_factory=dict)  # "kind:category" -> count
    final_failures: dict[str, int] = field(default_factory=dict)        # "kind:category" -> count
    retries: int = 0
    retry_sleep_s: float = 0.0
    embed_items_requested: int = 0     # texts asked for (incl. cache hits)
    embed_items_from_cache: int = 0
    embed_items_embedded: int = 0      # texts embedded by successful API calls
    embed_items_attempted: int = 0     # texts sent in all attempts, incl. failed ones
    embed_chars_sent: int = 0
    batch_fallbacks: int = 0
    tokens: dict[str, int] = field(default_factory=dict)                # summed usage_metadata fields
    pacing_sleep_s: float = 0.0
    errors: list[dict] = field(default_factory=list)                    # bounded log of failed attempts

    def bump(self, table: dict, key: str, n: int = 1) -> None:
        table[key] = table.get(key, 0) + n

    def record_error(self, kind: str, attempt: int, exc: BaseException, wait_s: float | None) -> None:
        if len(self.errors) < MAX_ERROR_LOG:
            self.errors.append({
                "kind": kind, "attempt": attempt, "category": classify_error(exc), **error_details(exc),
                "retry_after_s": retry_after_seconds(exc), "quota": quota_info(exc), "waited_s": wait_s,
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class RetryBudgetExhausted(RuntimeError):
    """Raised when a call still fails after the bounded retry budget."""

    def __init__(self, kind: str, category: str, attempts: int, cause: BaseException):
        super().__init__(f"{kind}: {category} after {attempts} attempt(s)")
        self.kind, self.category, self.attempts, self.cause = kind, category, attempts, cause
        self.quota = quota_info(cause)
        self.retry_after_s = retry_after_seconds(cause)
        self.details = error_details(cause)


class DailyQuotaExhausted(RetryBudgetExhausted):
    """A daily (or other long-horizon) quota is exhausted: retrying within the
    run is pointless, so the call fails on the first attempt."""


# --- error inspection -------------------------------------------------------------------
LONG_HORIZON_RETRY_S = 600.0  # a server delay longer than this is not a per-minute limit


def is_daily_quota(exc: BaseException) -> bool:
    """429 whose QuotaFailure names a per-day quota, or whose retry delay is
    far beyond any per-minute window (e.g. 'Please retry in 16h30m39s')."""
    if any("perday" in str(q.get("quotaId", "")).lower() for q in quota_info(exc) or []):
        return True
    delay = retry_after_seconds(exc)
    return delay is not None and delay > LONG_HORIZON_RETRY_S


def classify_error(exc: BaseException) -> str:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        if code == 429:
            return "daily_quota" if is_daily_quota(exc) else "rate_limit"
        if code >= 500:
            return "server"
        return f"client_{code}"
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    return "other"


MAX_ERROR_MESSAGE = 500
_SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),                                        # Google API keys
    re.compile(r"(?i)\b(key|api[_-]?key|access_token|token)=[^&\s\"']+"),          # credentials in URLs
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"),                                # bearer tokens
)


def sanitize_message(text: str) -> str:
    """Redact anything shaped like a credential and bound the length."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    text = " ".join(text.split())
    return text if len(text) <= MAX_ERROR_MESSAGE else text[:MAX_ERROR_MESSAGE] + "…"


def error_details(exc: BaseException) -> dict:
    """Exception type, HTTP status code, Google status string and the sanitized
    server message. Only the provider's error text is kept — never the request
    (prompt, context or paper contents)."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    payload = _error_payload(exc)
    message = payload.get("message") or getattr(exc, "message", None) or str(exc)
    status = getattr(exc, "status", None) or payload.get("status")
    return {
        "type": type(exc).__name__,
        "http_status": code if isinstance(code, int) else None,
        "status": str(status) if status else None,
        "message": sanitize_message(str(message or "")),
    }


def _retryable(category: str) -> bool:
    return category in {"rate_limit", "server", "timeout"}


def _error_payload(exc: BaseException) -> dict:
    details = getattr(exc, "details", None)
    if isinstance(details, list) and len(details) == 1:
        details = details[0]
    if isinstance(details, dict):
        return details.get("error", details)
    return {}


_DURATION = re.compile(r"^\s*(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:(\d+(?:\.\d+)?)s)?\s*$")
_RETRY_IN_MESSAGE = re.compile(r"retry in\s+((?:\d+h)?\s*(?:\d+m(?!s))?\s*(?:\d+(?:\.\d+)?s)?)", re.IGNORECASE)


def parse_duration(text: str) -> float | None:
    """'25s', '25.2s', '1m5s', '16h30m39.29s' -> seconds; None if unparseable."""
    match = _DURATION.match(text or "")
    if not match or not any(match.groups()):
        return None
    h, m, s = match.groups()
    return int(h or 0) * 3600 + int(m or 0) * 60 + float(s or 0)


def retry_after_seconds(exc: BaseException) -> float | None:
    """Server-requested delay: google.rpc.RetryInfo.retryDelay ('25s'), else
    'Please retry in 16h30m39.2s.' in the message. None if absent."""
    payload = _error_payload(exc)
    for item in payload.get("details", []) or []:
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("google.rpc.RetryInfo"):
            seconds = parse_duration(str(item.get("retryDelay", "")))
            if seconds is not None:
                return seconds
    message = str(payload.get("message", "")) or str(getattr(exc, "message", "") or "")
    match = _RETRY_IN_MESSAGE.search(message)
    return parse_duration(match.group(1)) if match else None


def quota_info(exc: BaseException) -> list[dict] | None:
    """quotaId / quotaMetric / quotaValue from google.rpc.QuotaFailure, if any."""
    payload = _error_payload(exc)
    found = []
    for item in payload.get("details", []) or []:
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("google.rpc.QuotaFailure"):
            for v in item.get("violations", []) or []:
                found.append({k: v.get(k) for k in ("quotaId", "quotaMetric", "quotaValue", "quotaDimensions") if k in v})
    return found or None


def _backoff_seconds(attempt: int) -> float:
    """Bounded exponential back-off with jitter (attempt = 1, 2, ...)."""
    return min(MAX_SINGLE_WAIT_S, 2.0 ** attempt) + random.uniform(0, 1.0)


def retry_wait_seconds(attempt: int, exc: BaseException) -> float:
    """Wait at least the server's requested delay + buffer, never less than
    the exponential back-off, never more than MAX_SINGLE_WAIT_S."""
    wait = _backoff_seconds(attempt)
    server = retry_after_seconds(exc)
    if server is not None:
        wait = max(wait, server + RETRY_BUFFER_S)
    return min(wait, MAX_SINGLE_WAIT_S)


# --- pacing -------------------------------------------------------------------------------
class Pacer:
    """Minimum interval between requests of one kind (client-side RPM cap)."""

    def __init__(self, rpm: float, stats: CallStats):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self.stats = stats
        self._last = 0.0

    def wait(self) -> None:
        if self.interval:
            delay = self._last + self.interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
                self.stats.pacing_sleep_s += delay
        self._last = time.monotonic()

    async def await_turn(self) -> None:
        if self.interval:
            delay = self._last + self.interval - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
                self.stats.pacing_sleep_s += delay
        self._last = time.monotonic()


class ItemPacer:
    """Sliding-window budget of embedded *items* per window (default 60 s).

    Every attempt — successful or not — consumes its items, because a rejected
    request may still count against the provider's quota."""

    def __init__(self, items_per_window: int, stats: CallStats, window_s: float = 60.0,
                 clock=time.monotonic, sleep=time.sleep):
        self.limit = int(items_per_window or 0)
        self.window_s, self.stats, self.clock, self.sleep = window_s, stats, clock, sleep
        self.events: deque[tuple[float, int]] = deque()

    def used(self) -> int:
        now = self.clock()
        while self.events and self.events[0][0] <= now - self.window_s:
            self.events.popleft()
        return sum(n for _, n in self.events)

    def acquire(self, n: int) -> None:
        if not self.limit:
            return
        if n > self.limit:
            raise ValueError(f"batch of {n} items exceeds the per-window budget of {self.limit}")
        while self.used() + n > self.limit:
            wait = self.events[0][0] + self.window_s - self.clock() + 0.05
            if wait > 0:
                self.sleep(wait)
                self.stats.pacing_sleep_s += wait
        self.events.append((self.clock(), n))


def make_client():
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (export it in your shell; it is never logged)")
    return genai.Client(api_key=api_key)


# --- embeddings ---------------------------------------------------------------------------
class Embedder:
    def __init__(self, client, *, model: str, dim: int, stats: CallStats, conn=None, batch_size: int = 45,
                 mode: str = "batch", items_per_min: int = 0, max_retries: int = 4, sleep=None):
        if mode not in {"batch", "single"}:
            raise ValueError("mode must be 'batch' or 'single'")
        self.client, self.model, self.dim, self.stats, self.conn = client, model, dim, stats, conn
        self.mode, self.max_retries = mode, max_retries
        self.sleep = sleep or time.sleep
        self.pacer = ItemPacer(items_per_min, stats, sleep=self.sleep)
        limit = self.pacer.limit or batch_size
        self.batch_size = max(1, min(batch_size, limit))  # a batch can never exceed the item budget

    def _config(self):
        from google.genai import types

        # Identical to production: only output_dimensionality, no task_type.
        return types.EmbedContentConfig(output_dimensionality=self.dim)

    def _call(self, kind: str, contents, n_items: int):
        attempt = 0
        while True:
            self.pacer.acquire(n_items)
            self.stats.bump(self.stats.attempts, kind)
            self.stats.embed_items_attempted += n_items
            started = time.perf_counter()
            try:
                response = self.client.models.embed_content(model=self.model, contents=contents, config=self._config())
                self.stats.bump(self.stats.requests, kind)
                self.stats.embed_items_embedded += n_items
                return response, time.perf_counter() - started
            except Exception as exc:  # noqa: BLE001 — classified, retried or re-raised below
                category = classify_error(exc)
                if attempt == 0:
                    self.stats.bump(self.stats.first_attempt_errors, f"{kind}:{category}")
                if not _retryable(category) or attempt >= self.max_retries:
                    self.stats.record_error(kind, attempt + 1, exc, None)
                    self.stats.bump(self.stats.final_failures, f"{kind}:{category}")
                    if category == "daily_quota":
                        raise DailyQuotaExhausted(kind, category, attempt + 1, exc) from exc
                    if _retryable(category):
                        raise RetryBudgetExhausted(kind, category, attempt + 1, exc) from exc
                    raise
                attempt += 1
                wait = retry_wait_seconds(attempt, exc)
                self.stats.record_error(kind, attempt, exc, wait)
                d = error_details(exc)
                log.warning("%s attempt %d failed (%s, HTTP %s %s: %s) — waiting %.1fs before retry %d/%d",
                            kind, attempt, category, d["http_status"], d["status"], d["message"],
                            wait, attempt, self.max_retries)
                self.stats.retries += 1
                self.stats.retry_sleep_s += wait
                self.sleep(wait)

    def embed_one(self, text: str) -> list[float]:
        """Exactly production's call shape: contents=<str>."""
        response, _ = self._call("embed_single", text, 1)
        self.stats.embed_chars_sent += len(text)
        return list(response.embeddings[0].values)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        contents = [types.Content(role="user", parts=[types.Part(text=t)]) for t in texts]
        response, _ = self._call("embed_batch", contents, len(texts))
        vectors = [list(e.values) for e in (response.embeddings or [])]
        if len(vectors) != len(texts):
            log.warning("batch returned %d embeddings for %d inputs — falling back to per-item calls",
                        len(vectors), len(texts))
            self.stats.batch_fallbacks += 1
            return [self.embed_one(t) for t in texts]
        self.stats.embed_chars_sent += sum(len(t) for t in texts)
        return vectors

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        from db import cached_embeddings, store_embeddings, text_sha256

        self.stats.embed_items_requested += len(texts)
        cache = cached_embeddings(self.conn, self.model, self.dim, texts) if self.conn is not None else {}
        missing, seen = [], set()
        for t in texts:
            h = text_sha256(t)
            if h not in cache and h not in seen:
                missing.append(t)
                seen.add(h)
        self.stats.embed_items_from_cache += len(texts) - len(missing)
        fresh: list[tuple[str, list[float]]] = []
        if self.mode == "single":
            for t in missing:
                fresh.append((t, self.embed_one(t)))
                if self.conn is not None:
                    store_embeddings(self.conn, self.model, self.dim, fresh[-1:])
        else:
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start:start + self.batch_size]
                pairs = list(zip(batch, self._embed_batch(batch)))
                fresh.extend(pairs)
                if self.conn is not None:
                    store_embeddings(self.conn, self.model, self.dim, pairs)  # cached per batch: resumable
        for t, v in fresh:
            cache[text_sha256(t)] = v
        for vec in cache.values():
            if len(vec) != self.dim:
                raise RuntimeError(f"embedding has {len(vec)} dims, expected {self.dim}")
        return [cache[text_sha256(t)] for t in texts]

    def embed_query(self, text: str) -> tuple[list[float], float, bool]:
        """Production embeds the query with a single call. Returns (vector, seconds, cache_hit)."""
        from db import cached_embeddings, store_embeddings, text_sha256

        if self.conn is not None:
            hit = cached_embeddings(self.conn, self.model, self.dim, [text]).get(text_sha256(text))
            if hit is not None:
                return hit, 0.0, True
        response, seconds = self._call("embed_query", text, 1)
        self.stats.embed_chars_sent += len(text)
        vec = list(response.embeddings[0].values)
        if self.conn is not None:
            store_embeddings(self.conn, self.model, self.dim, [(text, vec)])
        return vec, seconds, False


# --- generation -------------------------------------------------------------------------------
USAGE_FIELDS = ("prompt_token_count", "candidates_token_count", "thoughts_token_count",
                "cached_content_token_count", "total_token_count")


def usage_to_dict(usage) -> dict:
    if usage is None:
        return {}
    out = {}
    for name in USAGE_FIELDS:
        value = getattr(usage, name, None)
        if value is not None:
            out[name] = int(value)
    return out


async def stream_generate(client, *, model: str, contents, system_instruction: str, stats: CallStats,
                          pacer: Pacer, max_retries: int = 4) -> dict:
    """Production's call: client.aio.models.generate_content_stream(..., system_instruction=...).

    Returns text, time-to-first-token, total seconds, usage, attempts and error info.
    A retry only happens if the failure occurred before any text arrived, and
    waits at least the server's requested delay.
    """
    from google.genai import types

    config = types.GenerateContentConfig(system_instruction=system_instruction)
    attempt = 0
    while True:
        await pacer.await_turn()
        stats.bump(stats.attempts, "generate")
        started = time.perf_counter()
        text, ttft, usage = "", None, None
        try:
            stream = await client.aio.models.generate_content_stream(model=model, contents=contents, config=config)
            async for chunk in stream:
                if getattr(chunk, "usage_metadata", None) is not None:
                    usage = chunk.usage_metadata
                if chunk.text:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    text += chunk.text
            stats.bump(stats.requests, "generate")
            usage_dict = usage_to_dict(usage)
            for k, v in usage_dict.items():
                stats.bump(stats.tokens, k, v)
            return {"text": text, "ttft_s": ttft, "total_s": time.perf_counter() - started,
                    "usage": usage_dict, "attempts": attempt + 1, "error": None}
        except Exception as exc:  # noqa: BLE001
            category = classify_error(exc)
            if attempt == 0:
                stats.bump(stats.first_attempt_errors, f"generate:{category}")
            if text or not _retryable(category) or attempt >= max_retries:
                stats.record_error("generate", attempt + 1, exc, None)
                stats.bump(stats.final_failures, f"generate:{category}")
                usage_dict = usage_to_dict(usage)
                for k, v in usage_dict.items():
                    stats.bump(stats.tokens, k, v)
                return {"text": text, "ttft_s": ttft, "total_s": time.perf_counter() - started,
                        "usage": usage_dict, "attempts": attempt + 1,
                        "error": {"category": category, **error_details(exc),
                                  "retry_budget_exhausted": _retryable(category) and not text,
                                  "daily_quota": category == "daily_quota",
                                  # text had already streamed: the answer is incomplete, never a success
                                  "partial_answer": bool(text),
                                  "quota": quota_info(exc), "retry_after_s": retry_after_seconds(exc)}}
            attempt += 1
            wait = retry_wait_seconds(attempt, exc)
            stats.record_error("generate", attempt, exc, wait)
            d = error_details(exc)
            log.warning("generate attempt %d failed (%s, HTTP %s %s: %s) — waiting %.1fs before retry %d/%d",
                        attempt, category, d["http_status"], d["status"], d["message"], wait, attempt, max_retries)
            stats.retries += 1
            stats.retry_sleep_s += wait
            await asyncio.sleep(wait)
