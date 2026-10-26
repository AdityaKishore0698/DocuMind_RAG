"""Migration 003_embedding_space — record which embedding model produced each document's vectors.

    python scripts/migrate.py up --database-url URL                                  # adds the column
    python scripts/migrate.py up --database-url URL --assume-embedding-model gemini:gemini-embedding-2

Vectors from different embedding models are not comparable even when their dimensions match. After this
migration every document stores its embedding space (`documents.embedding_model`, e.g.
`ollama:nomic-embed-text` or `gemini:gemini-embedding-2`) and retrieval only searches documents whose space
equals the running configuration. Documents ingested before this migration have NULL — an unknown space —
and are excluded until they are re-processed, unless the operator knows which model produced them and labels
them with `--assume-embedding-model` (only rows that are still NULL are touched). Additive, idempotent;
`down` drops the column.
"""

from __future__ import annotations

NAME = "003_embedding_space"
LOCK_TIMEOUT = "5s"


def _ddl(cur, sql: str) -> None:
    cur.execute(f"SET lock_timeout = '{LOCK_TIMEOUT}'")
    cur.execute(sql)


def status(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name = 'documents' "
                    "AND column_name = 'embedding_model'")
        has_column = cur.fetchone() is not None
        unlabelled = None
        if has_column:
            cur.execute("SELECT count(*) FROM documents WHERE embedding_model IS NULL")
            unlabelled = cur.fetchone()[0]
        cur.execute("SELECT to_regclass('documind_migrations')")
        recorded = False
        if cur.fetchone()[0] is not None:
            cur.execute("SELECT 1 FROM documind_migrations WHERE name = %s", (NAME,))
            recorded = cur.fetchone() is not None
    return {"column": has_column, "unlabelled_documents": unlabelled, "recorded": recorded, "complete": has_column}


def up(conn, log=print, assume_embedding_model: str | None = None, **_) -> dict:
    with conn.cursor() as cur:
        _ddl(cur, "ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding_model text")
        log("documents.embedding_model in place")
        if assume_embedding_model:
            cur.execute("UPDATE documents SET embedding_model = %s WHERE embedding_model IS NULL",
                        (assume_embedding_model,))
            log(f"labelled {cur.rowcount} existing document(s) as {assume_embedding_model}")
        cur.execute("CREATE TABLE IF NOT EXISTS documind_migrations (name text PRIMARY KEY, applied_at timestamptz "
                    "NOT NULL DEFAULT now())")
        cur.execute("INSERT INTO documind_migrations (name) VALUES (%s) ON CONFLICT DO NOTHING", (NAME,))
    result = status(conn)
    log(f"up complete: {result}")
    return result


def down(conn, log=print) -> dict:
    with conn.cursor() as cur:
        _ddl(cur, "ALTER TABLE documents DROP COLUMN IF EXISTS embedding_model")
        cur.execute("SELECT to_regclass('documind_migrations')")
        if cur.fetchone()[0] is not None:
            cur.execute("DELETE FROM documind_migrations WHERE name = %s", (NAME,))
    result = status(conn)
    log(f"down complete: {result}")
    return result
