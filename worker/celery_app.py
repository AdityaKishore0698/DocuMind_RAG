import os
from celery import Celery


def redis_url() -> str:
    """REDIS_URL from the environment, made safe for Celery/kombu.

    kombu rejects a ``rediss://`` (TLS) broker URL that has no explicit
    ``ssl_cert_reqs`` parameter — Upstash URLs come without one.
    """
    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    if url.startswith("rediss://") and "ssl_cert_reqs" not in url:
        url += ("&" if "?" in url else "?") + "ssl_cert_reqs=required"
    return url


_url = redis_url()

celery_app = Celery(
    "worker",
    broker=_url,
    backend=_url,
    include=["tasks"]
)
