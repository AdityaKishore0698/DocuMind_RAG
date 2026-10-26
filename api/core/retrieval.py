"""Chunk retrieval for /chat: vector-only (default) or BM25 hybrid behind ``RETRIEVAL_MODE``.

Scope (both modes): the requesting user's papers **in the current chat** (papers are chat-scoped, migration 004)
whose vectors are in the active embedding space (migration 003).

* ``vector`` (default) — the original query plus the scope filters: chunks ordered by pgvector cosine distance, top 5.
* ``hybrid_bm25`` — one SQL statement: vector top-20 and BM25 top-20 over the scoped chunks, fused with
  Reciprocal Rank Fusion (k = 60). BM25 statistics (N, avgdl, df) are computed in the query over **the scoped chunks
  only**, from columns maintained by the ``001_bm25_hybrid`` migration (scripts/migrations/bm25.py), so they can never
  be stale and no other tenant influences a user's ranking. Parameters are fixed (k1 1.2, b 0.75) and match the
  evaluated configuration (evaluation/hybrid.py, docs/proposals/bm25-hybrid-production.md).

Hybrid runs inside a SAVEPOINT with ``set_config('statement_timeout', …, is_local => true)``. On timeout or any
database error the savepoint is rolled back (which also undoes the timeout) and the vector query runs instead. Every
fallback emits one structured log line and increments a per-reason counter; neither contains the question or any
document text.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from typing import Optional, Sequence

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from core.inference import embedding_space
from models.document import Document, DocumentChunk

log = logging.getLogger("documind.retrieval")

MODES = ("vector", "hybrid_bm25")
TOP_K = 5  # default; RETRIEVAL_TOP_K overrides (plan §7.2: benchmark 5 vs 10 rather than assume)
CANDIDATES = 20
RRF_K = 60
BM25_K1 = 1.2
BM25_B = 0.75
REQUIRED_COLUMNS = {"content_tsv", "token_count"}

FALLBACKS: Counter = Counter()  # reason -> count, since process start


class RetrievalConfigError(RuntimeError):
    pass


def retrieval_mode() -> str:
    mode = os.getenv("RETRIEVAL_MODE", "vector").strip().lower()
    if mode not in MODES:
        raise RetrievalConfigError(f"RETRIEVAL_MODE must be one of {MODES}, got {mode!r}")
    return mode


def default_top_k() -> int:
    value = int(os.getenv("RETRIEVAL_TOP_K", str(TOP_K)))
    if not 1 <= value <= 20:
        raise RetrievalConfigError("RETRIEVAL_TOP_K must be between 1 and 20")
    return value


def hybrid_timeout_ms() -> int:
    value = int(os.getenv("RETRIEVAL_HYBRID_TIMEOUT_MS", "300"))
    if value <= 0:
        raise RetrievalConfigError("RETRIEVAL_HYBRID_TIMEOUT_MS must be positive")
    return value


def validate_configuration(engine) -> str:
    """Called at startup: an unknown mode, or hybrid mode without the migration, is a startup error."""
    mode = retrieval_mode()
    default_top_k()
    if mode == "hybrid_bm25":
        hybrid_timeout_ms()
        with engine.connect() as conn:
            columns = {r[0] for r in conn.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'document_chunks' "
                "AND column_name IN ('content_tsv', 'token_count')"))}
            trigger = conn.execute(text(
                "SELECT 1 FROM pg_trigger WHERE tgname = 'documind_chunks_fts' AND NOT tgisinternal")).first()
        if columns != REQUIRED_COLUMNS or trigger is None:
            raise RetrievalConfigError("RETRIEVAL_MODE=hybrid_bm25 requires migration 001_bm25_hybrid "
                                       "(scripts/migrations/bm25.py up)")
    log.info("retrieval mode: %s", mode)
    return mode


def fallback_counts() -> dict:
    return dict(FALLBACKS)


def _record_fallback(reason: str, started: float) -> None:
    FALLBACKS[reason] += 1
    # Structured, content-free: no question text, no document text.
    log.warning("event=retrieval_fallback reason=%s mode=hybrid_bm25 latency_ms=%.1f",
                reason, (time.perf_counter() - started) * 1000)


# --- vector (legacy, unchanged) ------------------------------------------------------------------------
def vector_chunks(db: Session, user_id: int, query_embedding: Sequence[float], top_k: int = TOP_K,
                  document_ids: Optional[Sequence[int]] = None, sections: Optional[Sequence[str]] = None,
                  space: Optional[str] = None, *, session_id: int):
    # RAG Vector Search: only chunks of papers that belong to the current user AND the current chat, embedded in
    # the same space as the query (vectors from another model are never compared, whatever their dimension).
    query = db.query(DocumentChunk).join(Document).filter(Document.user_id == user_id,
                                                          Document.session_id == session_id,
                                                          Document.embedding_model == (space or embedding_space()))
    if document_ids:  # an additional restriction, never a replacement for the user scope
        query = query.filter(DocumentChunk.document_id.in_(list(document_ids)))
    if sections:
        query = query.filter(DocumentChunk.section_heading.in_(list(sections)))
    return query.order_by(
        DocumentChunk.embedding.cosine_distance(query_embedding)
    ).limit(top_k).all()


# --- hybrid ----------------------------------------------------------------------------------------------
HYBRID_SQL = text("""
WITH qlex AS (
    SELECT DISTINCT lexeme FROM unnest(tsvector_to_array(to_tsvector('english', :query_text))) AS lexeme
), q AS (
    SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS tsq FROM qlex
), user_chunks AS (
    SELECT c.id, c.document_id, c.section_heading, c.embedding, c.content_tsv, c.token_count
    FROM document_chunks c JOIN documents d ON d.id = c.document_id
    WHERE d.user_id = :user_id AND d.session_id = :session_id AND d.embedding_model = :space
), stats AS (
    SELECT count(*) AS n_docs, avg(token_count)::float8 AS avgdl FROM user_chunks
), scoped AS (
    SELECT * FROM user_chunks
    WHERE (CAST(:document_ids AS integer[]) IS NULL OR document_id = ANY(CAST(:document_ids AS integer[])))
      AND (CAST(:sections AS text[]) IS NULL OR section_heading = ANY(CAST(:sections AS text[])))
), vec AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> CAST(:query_embedding AS vector), id) AS rnk
    FROM scoped ORDER BY embedding <=> CAST(:query_embedding AS vector), id LIMIT :candidates
), matched AS (
    SELECT uc.id, uc.document_id, uc.section_heading, uc.content_tsv, uc.token_count
    FROM user_chunks uc CROSS JOIN q
    WHERE q.tsq IS NOT NULL AND uc.content_tsv @@ q.tsq
), terms AS (
    SELECT m.id, m.document_id, m.section_heading, m.token_count, t.lexeme,
           coalesce(array_length(t.positions, 1), 1) AS tf
    FROM matched m CROSS JOIN LATERAL unnest(m.content_tsv) AS t JOIN qlex ON qlex.lexeme = t.lexeme
), df AS (
    SELECT lexeme, count(*) AS df FROM terms GROUP BY lexeme
), bm25 AS (
    SELECT terms.id,
           sum(ln(1 + (s.n_docs - df.df + 0.5) / (df.df + 0.5))
               * terms.tf * (:k1 + 1) / (terms.tf + :k1 * (1 - :b + :b * terms.token_count / s.avgdl))) AS score
    FROM terms JOIN df USING (lexeme) CROSS JOIN stats s
    WHERE (CAST(:document_ids AS integer[]) IS NULL OR terms.document_id = ANY(CAST(:document_ids AS integer[])))
      AND (CAST(:sections AS text[]) IS NULL OR terms.section_heading = ANY(CAST(:sections AS text[])))
    GROUP BY terms.id
), fts AS (
    SELECT id, row_number() OVER (ORDER BY score DESC, id) AS rnk FROM bm25 ORDER BY score DESC, id LIMIT :candidates
), fused AS (
    SELECT coalesce(v.id, f.id) AS id, v.rnk AS vector_rank, f.rnk AS fts_rank,
           coalesce(1.0 / (:rrf_k + v.rnk), 0) + coalesce(1.0 / (:rrf_k + f.rnk), 0) AS rrf_score
    FROM vec v FULL OUTER JOIN fts f ON v.id = f.id
)
SELECT fused.id, fused.vector_rank, fused.fts_rank, fused.rrf_score, (SELECT tsq IS NULL FROM q) AS no_lexemes
FROM fused
ORDER BY fused.rrf_score DESC,
         LEAST(coalesce(fused.vector_rank, 2147483647), coalesce(fused.fts_rank, 2147483647)),
         fused.id
LIMIT :top_k
""")


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in values) + "]"


def hybrid_ranking(db: Session, user_id: int, query_text: str, query_embedding: Sequence[float],
                   top_k: int = TOP_K, document_ids: Optional[Sequence[int]] = None,
                   sections: Optional[Sequence[str]] = None, space: Optional[str] = None, *, session_id: int) -> list:
    """Fused ranking rows (id, vector_rank, fts_rank, rrf_score, no_lexemes). Raises on database errors."""
    params = {"query_text": query_text, "user_id": user_id, "session_id": session_id,
              "query_embedding": _vector_literal(query_embedding),
              "space": space or embedding_space(),
              "document_ids": list(document_ids) if document_ids else None,
              "sections": list(sections) if sections else None, "candidates": CANDIDATES,
              "k1": BM25_K1, "b": BM25_B, "rrf_k": RRF_K, "top_k": top_k}
    return db.execute(HYBRID_SQL, params).all()


def retrieve_chunks(db: Session, *, user_id: int, session_id: int, query_text: str, query_embedding: Sequence[float],
                    top_k: Optional[int] = None, document_ids: Optional[Sequence[int]] = None,
                    sections: Optional[Sequence[str]] = None):
    """Top-k DocumentChunk rows for the user, in rank order, according to RETRIEVAL_MODE.

    The scope is always the user's papers in chat `session_id` (papers are chat-scoped) in the active embedding
    space; `document_ids` (already ownership-checked by the caller) and `sections` only narrow it further."""
    top_k = top_k or default_top_k()
    if retrieval_mode() == "vector":
        return vector_chunks(db, user_id, query_embedding, top_k, document_ids, sections, session_id=session_id)
    started = time.perf_counter()
    try:
        with db.begin_nested():  # SAVEPOINT: a cancelled/failed statement must not poison the request's transaction
            previous = db.execute(text("SHOW statement_timeout")).scalar()
            db.execute(text("SELECT set_config('statement_timeout', :t, true)"), {"t": f"{hybrid_timeout_ms()}ms"})
            rows = hybrid_ranking(db, user_id, query_text, query_embedding, top_k, document_ids, sections,
                                  session_id=session_id)
            db.execute(text("SELECT set_config('statement_timeout', :t, true)"), {"t": previous})
    except DBAPIError as exc:  # rolled back to the savepoint (this also undoes the local timeout)
        from psycopg2 import errors as pg_errors

        reason = "timeout" if isinstance(getattr(exc, "orig", None), pg_errors.QueryCanceled) else "error"
        _record_fallback(reason, started)
        return vector_chunks(db, user_id, query_embedding, top_k, document_ids, sections, session_id=session_id)
    if rows and rows[0].no_lexemes:
        _record_fallback("no_lexemes", started)  # lexical list empty → fused order equals vector order
    ids = [r.id for r in rows]
    by_id = {c.id: c for c in db.query(DocumentChunk).filter(DocumentChunk.id.in_(ids)).all()} if ids else {}
    return [by_id[i] for i in ids]
