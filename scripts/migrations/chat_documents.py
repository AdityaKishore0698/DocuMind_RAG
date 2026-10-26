"""Migration 004_chat_documents — every paper belongs to the chat it was uploaded in.

    python scripts/migrate.py up --database-url URL

Adds `documents.session_id` (nullable) with an index and a foreign key to `chat_sessions(id)` that cascades on
delete (deleting a chat deletes its papers; the API also removes their stored files). Additive and non-locking:
nullable ADD COLUMN, CREATE INDEX CONCURRENTLY, FK added NOT VALID then VALIDATE, short lock_timeout. Idempotent;
`down` drops the column.

Backfill: none. Papers uploaded before this migration keep `session_id = NULL` ("unassigned"). They are never
searched in any chat; the user can move each one into a chat explicitly (`POST /document/{id}/assign`). Assigning
them automatically (e.g. to the most recent chat) would silently put papers into conversations they were never
part of.
"""

from __future__ import annotations

NAME = "004_chat_documents"
LOCK_TIMEOUT = "5s"
INDEX = "ix_documents_session_id"
FK = "documents_session_id_fkey"


def _ddl(cur, sql: str) -> None:
    cur.execute(f"SET lock_timeout = '{LOCK_TIMEOUT}'")
    cur.execute(sql)


def _index_state(cur) -> str:
    cur.execute("""SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
                   WHERE c.relname = %s""", (INDEX,))
    row = cur.fetchone()
    return "missing" if row is None else ("valid" if row[0] else "invalid")


def _fk(cur):
    cur.execute("""SELECT con.convalidated, con.confdeltype FROM pg_constraint con
                   JOIN pg_class rel ON rel.oid = con.conrelid
                   JOIN pg_attribute att ON att.attrelid = con.conrelid AND att.attnum = ANY(con.conkey)
                   WHERE con.contype = 'f' AND rel.relname = 'documents' AND att.attname = 'session_id'""")
    return cur.fetchone()


def status(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name = 'documents' "
                    "AND column_name = 'session_id'")
        has_column = cur.fetchone() is not None
        unassigned = None
        if has_column:
            cur.execute("SELECT count(*) FROM documents WHERE session_id IS NULL")
            unassigned = cur.fetchone()[0]
        fk = _fk(cur) if has_column else None
        index = _index_state(cur)
        cur.execute("SELECT to_regclass('documind_migrations')")
        recorded = False
        if cur.fetchone()[0] is not None:
            cur.execute("SELECT 1 FROM documind_migrations WHERE name = %s", (NAME,))
            recorded = cur.fetchone() is not None
    fk_ok = bool(fk) and fk[0] and fk[1] == "c"
    return {"column": has_column, "index": index, "fk_cascade_valid": fk_ok, "unassigned_documents": unassigned,
            "recorded": recorded, "complete": has_column and index == "valid" and fk_ok}


def up(conn, log=print, **_) -> dict:
    with conn.cursor() as cur:
        _ddl(cur, "ALTER TABLE documents ADD COLUMN IF NOT EXISTS session_id integer")
        if _index_state(cur) == "invalid":
            _ddl(cur, f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX}")
        if _index_state(cur) == "missing":
            _ddl(cur, f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX} ON documents (session_id)")
        if _fk(cur) is None:
            _ddl(cur, f"ALTER TABLE documents ADD CONSTRAINT {FK} FOREIGN KEY (session_id) "
                      "REFERENCES chat_sessions (id) ON DELETE CASCADE NOT VALID")
        if not _fk(cur)[0]:  # validate separately: only a brief lock, rows are checked without blocking writes
            _ddl(cur, f"ALTER TABLE documents VALIDATE CONSTRAINT {FK}")
        log("documents.session_id, index and cascading foreign key in place (existing papers left unassigned)")
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
        _ddl(cur, "ALTER TABLE documents DROP COLUMN IF EXISTS session_id")  # drops its index and FK too
        cur.execute("SELECT to_regclass('documind_migrations')")
        if cur.fetchone()[0] is not None:
            cur.execute("DELETE FROM documind_migrations WHERE name = %s", (NAME,))
    result = status(conn)
    log(f"down complete: {result}")
    return result
