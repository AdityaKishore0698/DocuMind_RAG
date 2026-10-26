"""Structured, content-free request logging (plan Phase 5).

Every request gets an id (the caller's `X-Request-ID` if it is a short token, else a new one), returned in
the `X-Request-ID` response header and attached to every `documind.*` log line emitted while handling it.
Log lines are `key=value`; they carry ids, step names, counts, latencies, model names and error classes —
never prompts, document text, tokens or keys.
"""

from __future__ import annotations

import contextvars
import logging
import os
import re
import time
import uuid

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_VALID_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
log = logging.getLogger("documind.http")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging() -> None:
    logger = logging.getLogger("documind")
    if any(isinstance(f, RequestIdFilter) for h in logger.handlers for f in h.filters):
        return
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s level=%(levelname)s logger=%(name)s "
                                           "request_id=%(request_id)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False


async def request_logging_middleware(request, call_next):
    incoming = request.headers.get("x-request-id", "")
    rid = incoming if _VALID_ID.match(incoming) else uuid.uuid4().hex[:16]
    token = request_id_var.set(rid)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        log.info("event=request method=%s path=%s status=%s latency_ms=%.0f", request.method,
                 request.url.path, status, (time.perf_counter() - started) * 1000)
        request_id_var.reset(token)
