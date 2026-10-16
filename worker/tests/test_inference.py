"""worker/inference.py — no network (httpx.MockTransport), no database."""

import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inference  # noqa: E402


def test_default_provider_is_gemini(monkeypatch):
    monkeypatch.delenv("INFERENCE_PROVIDER", raising=False)
    assert inference.provider_name() == "gemini"


def test_ollama_document_embedding(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5436/documind_local")
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.25] * 768]})

    vec = inference._ollama_embed("chunk", transport=httpx.MockTransport(handler))
    assert len(vec) == 768 and seen["input"] == "search_document: chunk" and seen["model"] == "nomic-embed-text"


def test_ollama_refuses_shared_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@aws-0-eu.pooler.supabase.com:6543/postgres")
    with pytest.raises(inference.InferenceConfigError):
        inference._ollama_embed("chunk", transport=httpx.MockTransport(lambda r: httpx.Response(200)))


def test_ollama_unavailable(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:5436/documind_local")

    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(inference.InferenceUnavailable):
        inference._ollama_embed("chunk", transport=httpx.MockTransport(handler))
