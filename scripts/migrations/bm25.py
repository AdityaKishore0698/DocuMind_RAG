"""Staged, reversible migration for BM25 hybrid retrieval (docs/proposals/bm25-hybrid-production.md §3).

    python scripts/migrations/bm25.py up     --database-url postgresql://…@127.0.0.1:5435/db
    python scripts/migrations/bm25.py down   --database-url …
    python scripts/migrations/bm25.py status --database-url …

Local/staging only: any non-loopback host is refused (production needs a separate approval and a deliberate
change to this guard). Every DDL statement runs with a short lock_timeout; nothing rewrites the table.

up:   token-count function → nullable columns (metadata-only) → trigger → batched backfill
      → CREATE INDEX CONCURRENTLY (an INVALID index left by a failed attempt is dropped and rebuilt) → verify → ANALYZE
down: DROP INDEX CONCURRENTLY → drop trigger + functions → drop columns (metadata-only) → forget the migration
Both directions are idempotent.
"""

from __future__ import annotations

import argparse
import sys
import time
from urllib.parse import urlparse

import psycopg2

NAME = "001_bm25_hybrid"
LOCK_TIMEOUT = "5s"
BATCH = 1000
INDEX = "ix_document_chunks_content_tsv"
LOOPBACK = {"localhost", "127.0.0.1", "::1"}

TOKEN_COUNT_FN = """
CREATE OR REPLACE FUNCTION documind_token_count(tsvector) RETURNS integer
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS
$$ SELECT coalesce(sum(coalesce(array_length(positions, 1), 1)), 0)::integer FROM unnest($1) $$
"""
TRIGGER_FN = """
CREATE OR REPLACE FUNCTION documind_chunks_fts() RETURNS trigger LANGUAGE plpgsql AS
$$ BEGIN
     NEW.content_tsv := to_tsvector('english', coalesce(NEW.content, ''));
     NEW.token_count := documind_token_count(NEW.content_tsv);
     RETURN NEW;
   END $$
"""


def assert_local(url: str) -> None:
    host = urlparse(url).hostname
    if host not in LOOPBACK:
        raise SystemExit(f"refusing non-local database host {host!r}: production migration is not approved")


def connect(url: str):
    assert_local(url)
    conn = psycopg2.connect(url)
    conn.autocommit = True  # each step is its own transaction; CONCURRENTLY cannot run inside one
    return conn


def _ddl(cur, sql: str, params=None) -> None:
    cur.execute(f"SET lock_timeout = '{LOCK_TIMEOUT}'")
    cur.execute(sql, params)


def _index_state(cur) -> str:
    cur.execute("""SELECT i.indisvalid FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid
                   WHERE c.relname = %s""", (INDEX,))
    row = cur.fetchone()
    return "missing" if row is None else ("valid" if row[0] else "invalid")


def status(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("""SELECT column_name FROM information_schema.columns
                       WHERE table_name = 'document_chunks' AND column_name IN ('content_tsv', 'token_count')""")
        columns = sorted(r[0] for r in cur.fetchall())
        cur.execute("SELECT 1 FROM pg_trigger WHERE tgname = 'documind_chunks_fts' AND NOT tgisinternal")
        trigger = cur.fetchone() is not None
        nulls = None
        if len(columns) == 2:
            cur.execute("SELECT count(*) FROM document_chunks WHERE content_tsv IS NULL OR token_count IS NULL")
            nulls = cur.fetchone()[0]
        cur.execute("SELECT to_regclass('documind_migrations')")
        applied = False
        if cur.fetchone()[0] is not None:
            cur.execute("SELECT 1 FROM documind_migrations WHERE name = %s", (NAME,))
            applied = cur.fetchone() is not None
        return {"columns": columns, "trigger": trigger, "index": _index_state(cur), "null_rows": nulls,
                "recorded": applied,
                "complete": len(columns) == 2 and trigger and _index_state(cur) == "valid" and nulls == 0}


def up(conn, batch: int = BATCH, log=print, **_) -> dict:
    with conn.cursor() as cur:
        _ddl(cur, TOKEN_COUNT_FN)
        _ddl(cur, "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector, "
                  "ADD COLUMN IF NOT EXISTS token_count integer")
        _ddl(cur, TRIGGER_FN)
        _ddl(cur, "DROP TRIGGER IF EXISTS documind_chunks_fts ON document_chunks")
        _ddl(cur, "CREATE TRIGGER documind_chunks_fts BEFORE INSERT OR UPDATE OF content ON document_chunks "
                  "FOR EACH ROW EXECUTE FUNCTION documind_chunks_fts()")
        log("columns + trigger in place")
        filled, t0 = 0, time.perf_counter()
        while True:  # short transactions: one batch each (autocommit)
            cur.execute("""UPDATE document_chunks c
                           SET content_tsv = to_tsvector('english', coalesce(c.content, '')),
                               token_count = documind_token_count(to_tsvector('english', coalesce(c.content, '')))
                           WHERE c.id IN (SELECT id FROM document_chunks
                                          WHERE content_tsv IS NULL OR token_count IS NULL ORDER BY id LIMIT %s)""",
                        (batch,))
            if cur.rowcount == 0:
                break
            filled += cur.rowcount
        log(f"backfilled {filled} rows in {time.perf_counter() - t0:.2f}s")
        if _index_state(cur) == "invalid":
            log("dropping INVALID index left by an earlier failed build")
            _ddl(cur, f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX}")
        if _index_state(cur) == "missing":
            _ddl(cur, f"CREATE INDEX CONCURRENTLY {INDEX} ON document_chunks USING gin (content_tsv)")
        if _index_state(cur) != "valid":
            raise SystemExit("GIN index is not valid after CREATE INDEX CONCURRENTLY — re-run `up`")
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
        _ddl(cur, f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX}")
        _ddl(cur, "DROP TRIGGER IF EXISTS documind_chunks_fts ON document_chunks")
        _ddl(cur, "DROP FUNCTION IF EXISTS documind_chunks_fts()")
        _ddl(cur, "ALTER TABLE document_chunks DROP COLUMN IF EXISTS content_tsv, DROP COLUMN IF EXISTS token_count")
        _ddl(cur, "DROP FUNCTION IF EXISTS documind_token_count(tsvector)")
        cur.execute("SELECT to_regclass('documind_migrations')")
        if cur.fetchone()[0] is not None:
            cur.execute("DELETE FROM documind_migrations WHERE name = %s", (NAME,))
    result = status(conn)
    log(f"down complete: {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["up", "down", "status"])
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args()
    conn = connect(args.database_url)
    try:
        if args.command == "up":
            up(conn, args.batch)
        elif args.command == "down":
            down(conn)
        else:
            print(status(conn))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
