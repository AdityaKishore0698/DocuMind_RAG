"""Inference provider boundary: Gemini (default) or a local Ollama server.

``INFERENCE_PROVIDER`` selects the provider once per process:

* ``gemini`` (default) — the existing google-genai calls, unchanged.
* ``ollama``           — ``OLLAMA_CHAT_MODEL`` (required; no default — the model depends on the
  machine: ``qwen3:14b`` on the lab GPU, ``qwen3:8b`` on a Mac, see local_inference/) for
  generation and ``OLLAMA_EMBEDDING_MODEL`` (default ``nomic-embed-text``) for embeddings,
  served by the Ollama HTTP API at ``OLLAMA_BASE_URL`` (default ``http://127.0.0.1:11434``).

HTTP clients use ``trust_env=False``: the endpoint is a private tunnel, so system/corporate
HTTP proxy settings must not intercept it (observed: a proxy answered 127.0.0.1 requests).

There is deliberately NO fallback between providers: if Ollama is unreachable the
request fails with ``InferenceUnavailable``.

Embedding spaces are not interchangeable. Gemini and nomic-embed-text both return
768-dim vectors, so they fit the same ``Vector(768)`` column, but a query embedded by
one model cannot be compared with chunks embedded by the other. The Ollama provider
therefore refuses to run unless ``DATABASE_URL`` points at a loopback (local) database,
so local-model vectors can never be written into the shared/production database.

Nothing here logs prompts, document text or credentials.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Optional
from urllib.parse import urlparse

log = logging.getLogger("documind.inference")

EMBEDDING_DIM = 768  # Vector(768) in models/document.py
PROVIDERS = ("gemini", "ollama")
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

# nomic-embed-text is trained with task prefixes; queries and documents use different ones.
NOMIC_PREFIXES = {"query": "search_query: ", "document": "search_document: "}


class InferenceConfigError(RuntimeError):
    """Invalid or unsafe inference configuration."""


class InferenceUnavailable(RuntimeError):
    """The configured inference server could not be reached or rejected the request."""


Message = tuple[str, str]  # (role, text); role is "user" or "assistant"


# --- configuration ---------------------------------------------------------------------
@dataclass(frozen=True)
class OllamaSettings:
    base_url: str
    chat_model: str
    embedding_model: str
    connect_timeout_s: float
    read_timeout_s: float
    keep_alive: str
    num_ctx: int
    think: bool
    embedding_prefixes: bool


def provider_name() -> str:
    name = os.getenv("INFERENCE_PROVIDER", "gemini").strip().lower()
    if name not in PROVIDERS:
        raise InferenceConfigError(f"INFERENCE_PROVIDER must be one of {PROVIDERS}, got {name!r}")
    return name


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    if value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise InferenceConfigError(f"{name} must be a boolean, got {value!r}")


def ollama_settings() -> OllamaSettings:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "").strip()
    if not chat_model:
        raise InferenceConfigError("OLLAMA_CHAT_MODEL is required with INFERENCE_PROVIDER=ollama "
                                   "(e.g. qwen3:14b on the lab server, qwen3:8b on a Mac)")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise InferenceConfigError(f"OLLAMA_BASE_URL must be an http(s) URL, got {base_url!r}")
    try:
        settings = OllamaSettings(
            base_url=base_url,
            chat_model=chat_model,
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            connect_timeout_s=float(os.getenv("OLLAMA_CONNECT_TIMEOUT_S", "5")),
            read_timeout_s=float(os.getenv("OLLAMA_READ_TIMEOUT_S", "300")),  # includes model load
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "2m"),  # shared GPU: release VRAM soon after use
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
            think=_bool_env("OLLAMA_THINK", False),
            embedding_prefixes=_bool_env("OLLAMA_EMBEDDING_PREFIXES", True),
        )
    except ValueError as exc:
        raise InferenceConfigError(f"invalid Ollama setting: {exc}") from exc
    return settings


def assert_local_database(database_url: Optional[str]) -> None:
    """Local-model vectors must never be written to a shared database."""
    host = urlparse(database_url or "").hostname
    if host not in LOOPBACK_HOSTS:
        raise InferenceConfigError(
            "INFERENCE_PROVIDER=ollama requires DATABASE_URL on localhost — nomic-embed-text vectors "
            "are a different embedding space from the Gemini vectors in the shared database")


# --- Gemini (unchanged behaviour) --------------------------------------------------------
class GeminiProvider:
    name = "gemini"

    def __init__(self):
        from google import genai

        self._genai = genai
        self.chat_model = os.getenv("CHAT_MODEL", "gemini-3.6-flash")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
        self._client = None

    def client(self):
        # Instantiated lazily so the app (and test suite) can import without GEMINI_API_KEY set.
        if self._client is None:
            self._client = self._genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return self._client

    def embed(self, text: str, kind: str = "document") -> list[float]:
        from google.genai import types

        result = self.client().models.embed_content(
            model=self.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
        )
        return result.embeddings[0].values

    async def stream_chat(self, system_prompt: str, history: Iterable[Message], query: str) -> AsyncIterator[str]:
        from google.genai import types

        # Gemini expects "user" / "model" roles; our history stores "user" / "assistant".
        contents = [types.Content(role="model" if role == "assistant" else "user", parts=[types.Part(text=text)])
                    for role, text in history]
        contents.append(types.Content(role="user", parts=[types.Part(text=query)]))
        stream = await self.client().aio.models.generate_content_stream(
            model=self.chat_model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text


# --- Ollama -------------------------------------------------------------------------------
class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Optional[OllamaSettings] = None, transport=None):
        self.settings = settings or ollama_settings()
        self.chat_model = self.settings.chat_model
        self.embedding_model = self.settings.embedding_model
        self._transport = transport  # tests inject an httpx.MockTransport

    def _timeout(self):
        import httpx

        s = self.settings
        return httpx.Timeout(s.read_timeout_s, connect=s.connect_timeout_s)

    def _unavailable(self, what: str, exc: Exception) -> InferenceUnavailable:
        # Only the endpoint and error class are reported — never the request body.
        return InferenceUnavailable(f"Ollama {what} failed at {self.settings.base_url}: {type(exc).__name__}: {exc}")

    @staticmethod
    def _error_text(response) -> str:
        try:
            return str(response.json().get("error", ""))[:300]
        except ValueError:
            return response.text[:300]

    def _prefixed(self, text: str, kind: str) -> str:
        if not self.settings.embedding_prefixes:
            return text
        if kind not in NOMIC_PREFIXES:
            raise ValueError(f"kind must be 'query' or 'document', got {kind!r}")
        return NOMIC_PREFIXES[kind] + text

    def embed(self, text: str, kind: str = "document") -> list[float]:
        import httpx

        payload = {"model": self.embedding_model, "input": self._prefixed(text, kind),
                   "keep_alive": self.settings.keep_alive}
        try:
            with httpx.Client(timeout=self._timeout(), transport=self._transport, trust_env=False) as client:
                response = client.post(f"{self.settings.base_url}/api/embed", json=payload)
        except httpx.HTTPError as exc:
            raise self._unavailable("embedding", exc) from exc
        if response.status_code != 200:
            raise InferenceUnavailable(f"Ollama embedding returned HTTP {response.status_code}: "
                                       f"{self._error_text(response)}")
        vectors = response.json().get("embeddings") or []
        if len(vectors) != 1:
            raise InferenceUnavailable(f"Ollama returned {len(vectors)} embeddings for 1 input")
        vector = [float(x) for x in vectors[0]]
        if len(vector) != EMBEDDING_DIM:
            raise InferenceConfigError(f"{self.embedding_model} returned {len(vector)} dims; "
                                       f"the database column is Vector({EMBEDDING_DIM})")
        return vector

    async def stream_chat(self, system_prompt: str, history: Iterable[Message], query: str) -> AsyncIterator[str]:
        import httpx

        s = self.settings
        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": "assistant" if role == "assistant" else "user", "content": text}
                     for role, text in history]
        messages.append({"role": "user", "content": query})
        payload = {"model": s.chat_model, "messages": messages, "stream": True, "think": s.think,
                   "keep_alive": s.keep_alive, "options": {"num_ctx": s.num_ctx}}
        try:
            async with httpx.AsyncClient(timeout=self._timeout(), transport=self._transport, trust_env=False) as client:
                async with client.stream("POST", f"{s.base_url}/api/chat", json=payload) as response:
                    if response.status_code != 200:
                        await response.aread()
                        raise InferenceUnavailable(f"Ollama chat returned HTTP {response.status_code}: "
                                                   f"{self._error_text(response)}")
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        event = json.loads(line)
                        if event.get("error"):
                            raise InferenceUnavailable(f"Ollama chat error: {str(event['error'])[:300]}")
                        text = (event.get("message") or {}).get("content") or ""
                        if text:
                            yield text
                        if event.get("done"):
                            return
        except httpx.HTTPError as exc:
            raise self._unavailable("chat", exc) from exc


# --- selection ----------------------------------------------------------------------------
_provider = None


def get_provider():
    """The process-wide provider (validated on first use)."""
    global _provider
    if _provider is None:
        if provider_name() == "ollama":
            assert_local_database(os.getenv("DATABASE_URL"))
            _provider = OllamaProvider()
        else:
            _provider = GeminiProvider()
        log.info("inference provider: %s (chat=%s, embeddings=%s)",
                 _provider.name, _provider.chat_model, _provider.embedding_model)
    return _provider


def reset_provider() -> None:
    """For tests: forget the cached provider."""
    global _provider
    _provider = None
