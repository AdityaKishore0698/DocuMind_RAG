"""Reset the DocuMind schema in the database at DATABASE_URL.

DESTRUCTIVE: drops and recreates every table (the Supabase Auth switch changed
the `users` table). Reads DATABASE_URL from the environment.

    set -a && source .env && set +a
    python scripts/init_db.py
"""
import os
import sys

if not os.getenv("DATABASE_URL"):
    sys.exit("DATABASE_URL is not set. Run:  set -a && source .env && set +a")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api")))

from sqlalchemy import text  # noqa: E402
from core.database import engine, Base  # noqa: E402
import models.user  # noqa: E402,F401  registers users / chat_sessions / chat_messages
import models.document  # noqa: E402,F401  registers documents / document_chunks


def main() -> None:
    print("Dropping and recreating all DocuMind tables ...")
    Base.metadata.drop_all(bind=engine)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    print("OK — extension 'vector' + tables:", ", ".join(sorted(Base.metadata.tables)))


if __name__ == "__main__":
    main()
