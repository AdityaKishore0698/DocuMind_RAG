"""Embedding provider for ingestion: Gemini (default) or a local Ollama server.

Embedding-only counterpart of api/core/inference.py (the worker is a separate image).
``INFERENCE_PROVIDER=ollama`` embeds chunks with ``OLLAMA_EMBEDDING_MODEL`` (default
``nomic-embed-text``, document prefix ``search_document: ``). No fallback to Gemini,
and the Ollama path refuses a non-local DATABASE_URL: nomic vectors are a different
embedding space from the Gemini vectors stored in the shared database.
"""

import os
from urllib.parse import urlparse

EMBEDDING_DIM = 768
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class InferenceConfigError(RuntimeError):
    pass


class InferenceUnavailable(RuntimeError):
    pass


def provider_name() -> str:
    name = os.getenv("INFERENCE_PROVIDER", "gemini").strip().lower()
    if name not in ("gemini", "ollama"):
        raise InferenceConfigError(f"INFERENCE_PROVIDER must be 'gemini' or 'ollama', got {name!r}")
    return name


_genai_client = None


def _gemini_embed(text: str) -> list[float]:
    global _genai_client
    from google import genai
    from google.genai import types

    if _genai_client is None:
        _genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    result = _genai_client.models.embed_content(
        model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-2"),
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    return result.embeddings[0].values


def _ollama_embed(text: str, transport=None) -> list[float]:
    import httpx

    if urlparse(os.getenv("DATABASE_URL") or "").hostname not in LOOPBACK_HOSTS:
        raise InferenceConfigError("INFERENCE_PROVIDER=ollama requires DATABASE_URL on localhost "
                                   "(nomic-embed-text vectors must not enter the shared database)")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    prefix = "" if os.getenv("OLLAMA_EMBEDDING_PREFIXES", "true").lower() in {"0", "false", "no", "off"} \
        else "search_document: "
    timeout = httpx.Timeout(float(os.getenv("OLLAMA_READ_TIMEOUT_S", "300")),
                            connect=float(os.getenv("OLLAMA_CONNECT_TIMEOUT_S", "5")))
    payload = {"model": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"), "input": prefix + text,
               "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "2m")}
    try:
        # trust_env=False: the endpoint is a private tunnel; never route it via a system HTTP proxy.
        with httpx.Client(timeout=timeout, transport=transport, trust_env=False) as client:
            response = client.post(f"{base_url}/api/embed", json=payload)
    except httpx.HTTPError as exc:  # endpoint + error class only; never the chunk text
        raise InferenceUnavailable(f"Ollama embedding failed at {base_url}: {type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        raise InferenceUnavailable(f"Ollama embedding returned HTTP {response.status_code}")
    vectors = response.json().get("embeddings") or []
    if len(vectors) != 1 or len(vectors[0]) != EMBEDDING_DIM:
        raise InferenceConfigError(f"unexpected embedding shape from Ollama (expected 1 x {EMBEDDING_DIM})")
    return [float(x) for x in vectors[0]]


def get_embedding(text: str) -> list[float]:
    if provider_name() == "ollama":
        return _ollama_embed(text)
    return _gemini_embed(text)
