"""Startup check that the database has every column the ORM models map.

`Base.metadata.create_all` creates missing *tables* but never adds columns to existing ones, so a
database created before a migration would fail at the first query instead of at startup. This turns
that into an immediate, explicit error naming the missing columns and the command that fixes it.
"""

from __future__ import annotations

from sqlalchemy import inspect

from core.database import Base


class SchemaMismatch(RuntimeError):
    pass


def missing_columns(engine) -> list[str]:
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    missing = []
    for table in Base.metadata.sorted_tables:
        if table.name not in tables:
            missing.append(f"{table.name} (table)")
            continue
        present = {c["name"] for c in inspector.get_columns(table.name)}
        missing += [f"{table.name}.{c.name}" for c in table.columns if c.name not in present]
    return missing


def assert_schema_current(engine) -> None:
    missing = missing_columns(engine)
    if missing:
        raise SchemaMismatch("database schema is behind the application models — missing "
                             + ", ".join(missing)
                             + ". Apply migrations: python scripts/migrate.py up --database-url <url>")
