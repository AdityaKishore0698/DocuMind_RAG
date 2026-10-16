"""Local evaluation database: safety guard, schema, ingestion and retrieval.

The harness only ever talks to a *local* pgvector PostgreSQL named like
``documind_eval``. ``assert_local_eval_database`` refuses anything else, so a
mis-set variable cannot point a benchmark at Supabase/production.

The application tables are created from ``scripts/init_db.sql`` (the production
schema source of truth) and are left untouched by the harness's own bookkeeping,
which lives in separate ``eval_*`` tables.
"""

from __future__ import annotations

import hashlib
import os
from urllib.parse import urlparse

from common import REPO_ROOT

DEFAULT_EVAL_DATABASE_URL = "postgresql://postgres:postgres@127.0.0.1:5434/documind_eval"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
EVAL_USER_UID = "documind-eval-user"


class UnsafeDatabaseError(RuntimeError):
    """Raised when the configured URL is not a local evaluation database."""


def eval_database_url() -> str:
    return os.environ.get("EVAL_DATABASE_URL", DEFAULT_EVAL_DATABASE_URL)


def assert_local_eval_database(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    dbname = (parsed.path or "").lstrip("/")
    lowered = url.lower()
    if parsed.scheme not in {"postgresql", "postgres"}:
        raise UnsafeDatabaseError(f"unsupported scheme {parsed.scheme!r}")
    if "supabase" in lowered or "pooler" in lowered:
        raise UnsafeDatabaseError("refusing a Supabase/pooler URL — evaluation must use a local database")
    if host not in LOCAL_HOSTS:
        raise UnsafeDatabaseError(f"refusing non-local host {host!r} (allowed: {sorted(LOCAL_HOSTS)})")
    if "eval" not in dbname:
        raise UnsafeDatabaseError(f"database name {dbname!r} must contain 'eval' (e.g. documind_eval)")
    production_url = os.environ.get("DATABASE_URL")
    if production_url and production_url.strip() == url.strip():
        raise UnsafeDatabaseError("EVAL_DATABASE_URL must not equal DATABASE_URL")


def redacted(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.hostname or ""
    if parsed.port:
        netloc += f":{parsed.port}"
    return f"{parsed.scheme}://***@{netloc}{parsed.path}"


def connect(url: str | None = None):
    import psycopg2
    from pgvector.psycopg2 import register_vector

    url = url or eval_database_url()
    assert_local_eval_database(url)
    conn = psycopg2.connect(url, connect_timeout=10)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


EVAL_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS eval_embedding_cache (
    model        TEXT    NOT NULL,
    dim          INTEGER NOT NULL,
    text_sha256  TEXT    NOT NULL,
    embedding    VECTOR(768) NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (model, dim, text_sha256)
);
CREATE TABLE IF NOT EXISTS eval_chunk_provenance (
    chunk_id       INTEGER PRIMARY KEY REFERENCES document_chunks (id) ON DELETE CASCADE,
    system         TEXT    NOT NULL,
    paper_id       TEXT    NOT NULL,
    chunk_ordinal  INTEGER NOT NULL,
    char_start     INTEGER,
    char_end       INTEGER,
    pages          INTEGER[] NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_documents (
    document_id  INTEGER PRIMARY KEY REFERENCES documents (id) ON DELETE CASCADE,
    system       TEXT NOT NULL,
    paper_id     TEXT NOT NULL,
    pdf_sha256   TEXT NOT NULL,
    UNIQUE (system, paper_id)
);
"""


def init_schema(conn) -> None:
    """(Re)create the production app tables locally + harness tables.

    The embedding cache is preserved across resets so re-ingestion is free.
    """
    init_sql = (REPO_ROOT / "scripts" / "init_db.sql").read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS eval_chunk_provenance, eval_documents")
        cur.execute(init_sql)
        cur.execute(EVAL_TABLES_SQL)
    conn.commit()


def ensure_eval_user(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE supabase_uid = %s", (EVAL_USER_UID,))
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute(
            "INSERT INTO users (supabase_uid, email, username) VALUES (%s, NULL, 'eval') RETURNING id",
            (EVAL_USER_UID,),
        )
        user_id = cur.fetchone()[0]
    conn.commit()
    return user_id


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def vector_to_list(value) -> list[float]:
    """pgvector's psycopg2 adapter may return a ``pgvector.Vector`` (not iterable),
    a numpy array, or a plain sequence — normalize all of them."""
    if hasattr(value, "to_list"):
        value = value.to_list()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [float(x) for x in value]


def cached_embeddings(conn, model: str, dim: int, texts: list[str]) -> dict[str, list[float]]:
    hashes = list({text_sha256(t) for t in texts})
    if not hashes:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT text_sha256, embedding FROM eval_embedding_cache WHERE model = %s AND dim = %s AND text_sha256 = ANY(%s)",
            (model, dim, hashes),
        )
        return {h: vector_to_list(v) for h, v in cur.fetchall()}


def store_embeddings(conn, model: str, dim: int, pairs: list[tuple[str, list[float]]]) -> None:
    import numpy as np

    with conn.cursor() as cur:
        for text, vec in pairs:
            cur.execute(
                "INSERT INTO eval_embedding_cache (model, dim, text_sha256, embedding) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                (model, dim, text_sha256(text), np.asarray(vec, dtype=np.float32)),
            )
    conn.commit()


def paper_already_ingested(conn, system: str, paper_id: str, pdf_sha256: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pdf_sha256 FROM eval_documents WHERE system = %s AND paper_id = %s", (system, paper_id)
        )
        row = cur.fetchone()
    return bool(row and row[0] == pdf_sha256)


def insert_document_with_chunks(conn, *, system, user_id, paper, pdf_sha256, provenance, vectors) -> int:
    """Insert one paper the way production does: a documents row + one
    document_chunks row per chunk (status 'completed'), plus harness provenance."""
    import numpy as np

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (user_id, filename, status) VALUES (%s, %s, 'completed') RETURNING id",
            (user_id, paper.filename),
        )
        document_id = cur.fetchone()[0]
        for prov, vec in zip(provenance, vectors):
            cur.execute(
                "INSERT INTO document_chunks (document_id, content, embedding) VALUES (%s, %s, %s) RETURNING id",
                (document_id, prov.text, np.asarray(vec, dtype=np.float32)),
            )
            chunk_id = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO eval_chunk_provenance (chunk_id, system, paper_id, chunk_ordinal, char_start, char_end, pages) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (chunk_id, system, paper.id, prov.ordinal, prov.char_start, prov.char_end, prov.pages),
            )
        cur.execute(
            "INSERT INTO eval_documents (document_id, system, paper_id, pdf_sha256) VALUES (%s, %s, %s, %s)",
            (document_id, system, paper.id, pdf_sha256),
        )
    conn.commit()
    return document_id


def ingest_state_issues(conn, system: str, expected_chunks: dict[str, int]) -> dict:
    """Consistency check before (re)ingesting.

    Returns {"blocking": [...], "reingest": [paper_ids]}:
    * blocking  — rows the harness cannot attribute to a paper (documents with no
      eval_documents mapping, chunks with no provenance). These are never deleted
      automatically; the caller must stop.
    * reingest  — papers recorded as ingested whose stored chunk count differs
      from what System A's chunker produces now; safe to delete and redo.
    """
    blocking: list[str] = []
    with conn.cursor() as cur:
        cur.execute("SELECT d.id, d.filename FROM documents d LEFT JOIN eval_documents e ON e.document_id = d.id "
                    "WHERE e.document_id IS NULL")
        blocking += [f"document {i} ({f}) has no eval_documents mapping" for i, f in cur.fetchall()]
        cur.execute("SELECT count(*) FROM document_chunks c LEFT JOIN eval_chunk_provenance p ON p.chunk_id = c.id "
                    "WHERE p.chunk_id IS NULL")
        orphans = cur.fetchone()[0]
        if orphans:
            blocking.append(f"{orphans} document_chunks rows have no provenance")
        cur.execute("SELECT e.paper_id, count(c.id) FROM eval_documents e LEFT JOIN document_chunks c "
                    "ON c.document_id = e.document_id WHERE e.system = %s GROUP BY e.paper_id", (system,))
        stored = dict(cur.fetchall())
        cur.execute("SELECT paper_id, count(*) FROM eval_documents WHERE system = %s GROUP BY paper_id HAVING count(*) > 1",
                    (system,))
        blocking += [f"paper {p} is mapped {n} times" for p, n in cur.fetchall()]
    reingest = [p for p, n in stored.items() if p in expected_chunks and n != expected_chunks[p]]
    return {"blocking": blocking, "reingest": reingest, "stored": stored}


def delete_paper(conn, system: str, paper_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT document_id FROM eval_documents WHERE system = %s AND paper_id = %s", (system, paper_id))
        row = cur.fetchone()
        if row:
            # Production FKs have no ON DELETE CASCADE — delete children explicitly.
            cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (row[0],))
            cur.execute("DELETE FROM documents WHERE id = %s", (row[0],))
    conn.commit()


def retrieve(conn, sql: str, user_id: int, query_embedding: list[float], top_k: int) -> list[dict]:
    """Run a system's retrieval SQL and attach harness provenance for scoring."""
    import numpy as np

    with conn.cursor() as cur:
        cur.execute(sql, {"user_id": user_id, "query_embedding": np.asarray(query_embedding, dtype=np.float32), "top_k": top_k})
        rows = cur.fetchall()
        chunk_ids = [r[0] for r in rows]
        cur.execute(
            "SELECT chunk_id, paper_id, pages, chunk_ordinal FROM eval_chunk_provenance WHERE chunk_id = ANY(%s)",
            (chunk_ids,),
        )
        prov = {r[0]: (r[1], list(r[2]), r[3]) for r in cur.fetchall()}
    results = []
    for rank, (chunk_id, document_id, content, distance) in enumerate(rows, start=1):
        paper_id, pages, ordinal = prov.get(chunk_id, (None, [], None))
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "paper_id": paper_id,
                "pages": pages,
                "chunk_ordinal": ordinal,
                "cosine_distance": float(distance),
                "text": content,
            }
        )
    return results


def corpus_stats(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        docs = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM document_chunks")
        chunks = cur.fetchone()[0]
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
        cur.execute("SHOW server_version")
        server = cur.fetchone()[0]
    return {"documents": docs, "chunks": chunks, "pgvector": row[0] if row else None, "postgres": server}


def systems_present(conn) -> set[str]:
    """Systems whose documents are stored in this database (one embedding space each)."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('eval_documents')")
        if cur.fetchone()[0] is None:
            return set()
        cur.execute("SELECT DISTINCT system FROM eval_documents")
        return {row[0] for row in cur.fetchall()}
