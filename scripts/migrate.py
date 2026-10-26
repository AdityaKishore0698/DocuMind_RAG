"""Ordered schema migrations for ResearchLens (loopback databases only).

    python scripts/migrate.py status --database-url URL
    python scripts/migrate.py up     --database-url URL                     # required migrations
    python scripts/migrate.py up     --database-url URL 001_bm25_hybrid     # an optional one
    python scripts/migrate.py down   --database-url URL 002_provenance

| migration          | required | what it does                                                     |
|--------------------|----------|------------------------------------------------------------------|
| 001_bm25_hybrid    | no       | full-text columns, trigger, GIN index for RETRIEVAL_MODE=hybrid_bm25 |
| 002_provenance     | yes      | paper metadata, page/section/chunk-index provenance, message sources |
| 003_embedding_space| yes      | documents.embedding_model — which embedding model produced the vectors |
| 004_chat_documents | yes      | documents.session_id — each paper belongs to the chat it was uploaded in |

A database created by the API itself (fresh install) already has every column; `up` then only records.

The URL is always explicit (never read from the environment) and must be a loopback host.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent / "migrations"
MIGRATIONS = [("001_bm25_hybrid", "bm25.py", False), ("002_provenance", "provenance.py", True),
              ("003_embedding_space", "embedding_space.py", True), ("004_chat_documents", "chat_documents.py", True)]


def _load(filename: str):
    spec = importlib.util.spec_from_file_location(filename[:-3], HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modules() -> dict:
    return {name: (_load(fname), required) for name, fname, required in MIGRATIONS}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "up", "down"])
    parser.add_argument("names", nargs="*", help="migration names (default for up: all required ones)")
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--assume-embedding-model", default=None,
                        help="003: label documents ingested before the migration (e.g. gemini:gemini-embedding-2)")
    args = parser.parse_args()
    mods = modules()
    bm25 = mods["001_bm25_hybrid"][0]
    conn = bm25.connect(args.database_url)  # loopback-only guard, autocommit connection
    try:
        unknown = [n for n in args.names if n not in mods]
        if unknown:
            raise SystemExit(f"unknown migration(s): {unknown}")
        if args.command == "status":
            for name, (module, required) in mods.items():
                print(f"{name} ({'required' if required else 'optional'}): {module.status(conn)}")
        elif args.command == "up":
            names = args.names or [n for n, (_, req) in mods.items() if req]
            for name in [n for n, _, _ in MIGRATIONS if n in names]:
                print(f"== {name}")
                mods[name][0].up(conn, assume_embedding_model=args.assume_embedding_model)
        else:
            if not args.names:
                raise SystemExit("down needs explicit migration name(s)")
            for name in reversed([n for n, _, _ in MIGRATIONS if n in args.names]):
                print(f"== {name}")
                mods[name][0].down(conn)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
