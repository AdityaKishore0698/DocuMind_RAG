"""Health and startup diagnostics: database, Redis broker, inference server and embedding space.

Nothing here returns credentials or document content — only reachability, model names and counts.
"""

from __future__ import annotations

import logging
import os

from sqlalchemy import text

from core.inference import embedding_space, get_provider, ollama_settings, provider_name

log = logging.getLogger("documind.health")


def document_spaces(engine) -> dict[str, int]:
    """Completed documents per embedding space ("unknown" = ingested before migration 003)."""
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT coalesce(embedding_model, 'unknown'), count(*) FROM documents "
                                 "WHERE status = 'completed' GROUP BY 1")).all()
    return {space: count for space, count in rows}


def unassigned_documents(engine) -> int:
    """Papers uploaded before papers belonged to chats (migration 004): searched nowhere until moved into a chat."""
    with engine.connect() as conn:
        return conn.execute(text("SELECT count(*) FROM documents WHERE session_id IS NULL")).scalar() or 0


def log_embedding_spaces(engine) -> None:
    unassigned = unassigned_documents(engine)
    if unassigned:
        log.warning("event=unassigned_documents count=%s — uploaded before papers belonged to chats; users can move "
                    "each into a chat (they are not searched until then)", unassigned)
    space = embedding_space()
    others = {s: n for s, n in document_spaces(engine).items() if s != space}
    log.info("event=embedding_space active=%s", space)
    if others:
        log.warning("event=embedding_space_mismatch active=%s other_spaces=%s — these documents are excluded "
                    "from search until re-processed", space, others)


def _check_redis() -> dict:
    try:
        import redis

        client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"), socket_timeout=2,
                                      socket_connect_timeout=2)
        client.ping()
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__}


def _check_inference() -> dict:
    provider = get_provider()
    out = {"provider": provider_name(), "chat_model": provider.chat_model,
           "embedding_model": provider.embedding_model}
    if provider_name() != "ollama":
        out["ok"] = True  # hosted API: not probed (no paid/quota-consuming calls from a health check)
        return out
    import httpx

    settings = ollama_settings()
    try:
        with httpx.Client(timeout=3, trust_env=False) as client:
            tags = client.get(f"{settings.base_url}/api/tags").json()
        names = {m.get("name", "") for m in tags.get("models", [])}

        def present(model: str) -> bool:
            return model in names or f"{model}:latest" in names

        missing = [m for m in (settings.chat_model, settings.embedding_model) if not present(m)]
        out.update(ok=not missing, base_url=settings.base_url, missing_models=missing)
        if missing:
            out["hint"] = "ollama pull " + " ".join(missing)
    except Exception as exc:  # noqa: BLE001
        out.update(ok=False, base_url=settings.base_url, error=type(exc).__name__,
                   hint="is Ollama running and reachable at OLLAMA_BASE_URL?")
    return out


def health_report(engine) -> dict:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        database = {"ok": True, "documents_by_embedding_space": document_spaces(engine),
                    "unassigned_documents": unassigned_documents(engine)}
    except Exception as exc:  # noqa: BLE001
        database = {"ok": False, "error": type(exc).__name__}
    report = {"database": database, "redis": _check_redis(), "inference": _check_inference(),
              "embedding_space": embedding_space()}
    report["ok"] = all(part.get("ok") for part in (database, report["redis"], report["inference"]))
    return report
