"""Migration 002_provenance — paper metadata, chunk provenance, cited sources (plan Phase 1).

    python scripts/migrate.py up            # applies required migrations (this one) in order
    python scripts/migrate.py status

Additive and non-locking: nullable ADD COLUMN (metadata-only), CREATE INDEX CONCURRENTLY, and the
chunk foreign key is replaced with an ON DELETE CASCADE one via NOT VALID + VALIDATE (no long lock).
Every DDL runs with a short lock_timeout. Idempotent; `down` reverses it. A fresh database created by
the ORM (`create_all`) already matches, so `up` is then a no-op apart from recording itself.
"""

from __future__ import annotations

NAME = "002_provenance"
LOCK_TIMEOUT = "5s"
UNIQUE_INDEX = "uq_document_chunks_document_chunk_index"
DOC_INDEX = "ix_document_chunks_document_id"
CASCADE_FK = "document_chunks_document_id_cascade_fkey"

COLUMNS = {
    "documents": [("title", "text"), ("authors", "text"), ("year", "integer"), ("page_count", "integer"),
                  ("metadata", "jsonb"), ("error", "text")],
    "document_chunks": [("page_number", "integer"), ("section_heading", "text"), ("chunk_index", "integer")],
    "chat_messages": [("sources", "jsonb")],
}


def _ddl(cur, sql: str) -> None:
    cur.execute(f"SET lock_timeout = '{LOCK_TIMEOUT}'")
    cur.execute(sql)


def _index_state(cur, name: str) -> str:
    cur.execute("""SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
                   WHERE c.relname = %s""", (name,))
    row = cur.fetchone()
    return "missing" if row is None else ("valid" if row[0] else "invalid")


def _chunk_fks(cur) -> list[tuple[str, str]]:
    """(constraint name, on-delete action) of FKs from document_chunks.document_id to documents."""
    cur.execute("""
        SELECT con.conname, con.confdeltype FROM pg_constraint con
        JOIN pg_class rel ON rel.oid = con.conrelid
        JOIN pg_attribute att ON att.attrelid = con.conrelid AND att.attnum = ANY(con.conkey)
        WHERE con.contype = 'f' AND rel.relname = 'document_chunks' AND att.attname = 'document_id'""")
    return [(r[0], r[1]) for r in cur.fetchall()]


def status(conn) -> dict:
    with conn.cursor() as cur:
        missing = []
        for table, cols in COLUMNS.items():
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table,))
            present = {r[0] for r in cur.fetchall()}
            missing += [f"{table}.{c}" for c, _ in cols if c not in present]
        fks = _chunk_fks(cur)
        cur.execute("SELECT to_regclass('documind_migrations')")
        recorded = False
        if cur.fetchone()[0] is not None:
            cur.execute("SELECT 1 FROM documind_migrations WHERE name = %s", (NAME,))
            recorded = cur.fetchone() is not None
        result = {"missing_columns": missing, "unique_index": _index_state(cur, UNIQUE_INDEX),
                  "document_id_index": _index_state(cur, DOC_INDEX),
                  "fk_on_delete": sorted({action for _, action in fks}), "recorded": recorded}
    result["complete"] = (not missing and result["unique_index"] == "valid"
                          and result["document_id_index"] == "valid" and result["fk_on_delete"] == ["c"])
    return result


def _ensure_index(cur, name: str, sql: str, log) -> None:
    if _index_state(cur, name) == "invalid":
        log(f"dropping INVALID index {name} left by an earlier failed build")
        _ddl(cur, f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
    if _index_state(cur, name) == "missing":
        _ddl(cur, sql)
    if _index_state(cur, name) != "valid":
        raise SystemExit(f"index {name} is not valid — re-run `up`")


def up(conn, log=print, **_) -> dict:
    with conn.cursor() as cur:
        for table, cols in COLUMNS.items():
            adds = ", ".join(f"ADD COLUMN IF NOT EXISTS {c} {t}" for c, t in cols)
            _ddl(cur, f"ALTER TABLE {table} {adds}")
        log("columns in place")
        _ensure_index(cur, DOC_INDEX,
                      f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {DOC_INDEX} ON document_chunks (document_id)", log)
        _ensure_index(cur, UNIQUE_INDEX, f"CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS {UNIQUE_INDEX} "
                                         "ON document_chunks (document_id, chunk_index)", log)
        fks = _chunk_fks(cur)
        if not any(action == "c" for _, action in fks):
            _ddl(cur, f"ALTER TABLE document_chunks ADD CONSTRAINT {CASCADE_FK} FOREIGN KEY (document_id) "
                      "REFERENCES documents (id) ON DELETE CASCADE NOT VALID")
            _ddl(cur, f"ALTER TABLE document_chunks VALIDATE CONSTRAINT {CASCADE_FK}")
        for name, action in _chunk_fks(cur):
            if action != "c":
                _ddl(cur, f'ALTER TABLE document_chunks DROP CONSTRAINT "{name}"')
        log("chunk foreign key cascades on delete")
        cur.execute("ANALYZE document_chunks")
        cur.execute("CREATE TABLE IF NOT EXISTS documind_migrations (name text PRIMARY KEY, applied_at timestamptz "
                    "NOT NULL DEFAULT now())")
        cur.execute("INSERT INTO documind_migrations (name) VALUES (%s) ON CONFLICT DO NOTHING", (NAME,))
    result = status(conn)
    if not result["complete"]:
        raise SystemExit(f"migration incomplete: {result}")
    log(f"up complete: {result}")
    return result


def down(conn, log=print) -> dict:
    with conn.cursor() as cur:
        _ddl(cur, f"DROP INDEX CONCURRENTLY IF EXISTS {UNIQUE_INDEX}")
        fks = _chunk_fks(cur)
        if fks and not any(action != "c" for _, action in fks):
            _ddl(cur, "ALTER TABLE document_chunks ADD CONSTRAINT document_chunks_document_id_fkey FOREIGN KEY "
                      "(document_id) REFERENCES documents (id) NOT VALID")
            _ddl(cur, "ALTER TABLE document_chunks VALIDATE CONSTRAINT document_chunks_document_id_fkey")
            for name, action in fks:
                _ddl(cur, f'ALTER TABLE document_chunks DROP CONSTRAINT "{name}"')
        for table, cols in COLUMNS.items():
            drops = ", ".join(f"DROP COLUMN IF EXISTS {c}" for c, _ in cols)
            _ddl(cur, f"ALTER TABLE {table} {drops}")
        cur.execute("SELECT to_regclass('documind_migrations')")
        if cur.fetchone()[0] is not None:
            cur.execute("DELETE FROM documind_migrations WHERE name = %s", (NAME,))
    result = status(conn)
    log(f"down complete: {result}")
    return result
