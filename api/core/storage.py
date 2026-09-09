import os
import re

from supabase import Client, create_client

# Private Storage bucket that holds every uploaded document.
STORAGE_BUCKET = "documents"

_client: Client | None = None


def get_supabase() -> Client:
    """Lazily create a service-role Supabase client from the environment."""
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set"
            )
        _client = create_client(url, key)
    return _client


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def object_key(user_id: int, document_id: int, filename: str) -> str:
    """Deterministic Storage object key: ``<user_id>/<document_id>/<safe-name>``.

    Namespaced by user and document id so keys are unique and can be
    reconstructed later (e.g. on delete) without a DB column.
    """
    safe = _UNSAFE.sub("_", filename).strip("_") or "file"
    return f"{user_id}/{document_id}/{safe}"
