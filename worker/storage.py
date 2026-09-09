import os

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
