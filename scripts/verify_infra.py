"""Connectivity check for the managed infrastructure.

Verifies, using the values in the environment (load .env first):
  * Supabase Postgres  — server version, pgvector extension, a sample <=> query
  * Supabase Storage    — upload / download / delete in the 'documents' bucket
  * Upstash Redis       — PING + SET/GET/DEL over TLS

    set -a && source .env && set +a
    python scripts/verify_infra.py

Exit code is non-zero if any check fails.
"""
import os
import sys
import uuid

import psycopg2
import redis
from supabase import create_client

BUCKET = "documents"
ok = True


def check(name: str, fn) -> None:
    global ok
    try:
        detail = fn()
        print(f"  [ OK ] {name}: {detail}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] {name}: {e!r}")


def check_postgres() -> str:
    dsn = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(dsn)
    try:
        cur = conn.cursor()
        cur.execute("select version()")
        version = cur.fetchone()[0].split(" on ")[0]
        cur.execute("select extversion from pg_extension where extname = 'vector'")
        row = cur.fetchone()
        if not row:
            raise RuntimeError("pgvector extension not installed (run scripts/init_db.sql)")
        cur.execute("select ('[1,0,0]'::vector <=> '[0,1,0]'::vector)")
        distance = cur.fetchone()[0]
        cur.execute(
            "select count(*) from information_schema.tables "
            "where table_schema = 'public' and table_name = any(%s)",
            (["users", "documents", "document_chunks", "chat_sessions", "chat_messages"],),
        )
        table_count = cur.fetchone()[0]
        return f"{version}, pgvector {row[0]}, cosine<=> = {distance}, {table_count}/5 tables"
    finally:
        conn.close()


def check_storage() -> str:
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    bucket = client.storage.from_(BUCKET)
    key = f"_healthcheck/{uuid.uuid4().hex}.txt"
    payload = b"documind infra check"
    bucket.upload(key, payload, {"content-type": "text/plain", "upsert": "true"})
    got = bucket.download(key)
    bucket.remove([key])
    if got != payload:
        raise RuntimeError("downloaded bytes did not match uploaded bytes")
    return f"upload/download/delete round-trip on '{BUCKET}' bucket"


def check_redis() -> str:
    r = redis.from_url(os.environ["REDIS_URL"])
    if not r.ping():
        raise RuntimeError("PING returned falsy")
    key = f"documind:healthcheck:{uuid.uuid4().hex}"
    r.set(key, "ok", ex=30)
    value = r.get(key)
    r.delete(key)
    if value != b"ok":
        raise RuntimeError(f"GET returned {value!r}")
    return "PING + SET/GET/DEL over TLS"


if __name__ == "__main__":
    for required in ("DATABASE_URL", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "REDIS_URL"):
        if not os.getenv(required):
            sys.exit(f"{required} is not set. Run:  set -a && source .env && set +a")

    print("Verifying managed infrastructure...")
    check("Supabase Postgres + pgvector", check_postgres)
    check("Supabase Storage", check_storage)
    check("Upstash Redis", check_redis)
    print("All checks passed." if ok else "One or more checks FAILED.")
    sys.exit(0 if ok else 1)
