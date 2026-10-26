"""worker/inference.py — no network (fake batch functions / httpx.MockTransport), no database."""

import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inference  # noqa: E402


class ApiError(Exception):
    """Shaped like google.genai.errors.APIError (code + details)."""

    def __init__(self, code, retry=None, per_day=False):
        super().__init__(f"{code}")
        details = []
        if retry:
            details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry})
        if per_day:
            details.append({"violations": [{"quotaId": "EmbedContentRequestsPerDayPerProject"}]})
        self.code, self.details = code, {"error": {"code": code, "message": "x", "details": details}}


def test_default_provider_is_gemini(monkeypatch):
    monkeypatch.delenv("INFERENCE_PROVIDER", raising=False)
    assert inference.provider_name() == "gemini"


def test_batches_preserve_order_and_size(monkeypatch):
    monkeypatch.setenv("EMBED_BATCH_SIZE", "3")
    seen = []

    def batch(texts):
        seen.append(list(texts))
        return [[float(t)] for t in texts]

    out = inference.get_embeddings([str(i) for i in range(7)], batch_fn=batch, pacer=inference.ItemPacer(0))
    assert [len(b) for b in seen] == [3, 3, 1] and out == [[float(i)] for i in range(7)]


def test_rate_limit_is_retried_with_server_delay(monkeypatch):
    calls, sleeps = [], []

    def batch(texts):
        calls.append(1)
        if len(calls) < 3:
            raise ApiError(429, retry="7s")
        return [[1.0]] * len(texts)

    out = inference.get_embeddings(["a"], batch_fn=batch, pacer=inference.ItemPacer(0), sleep=sleeps.append)
    assert out == [[1.0]] and sleeps == [9.0, 9.0]


def test_retries_are_bounded():
    def batch(texts):
        raise ApiError(503)

    with pytest.raises(inference.InferenceUnavailable, match="server"):
        inference.get_embeddings(["a"], batch_fn=batch, pacer=inference.ItemPacer(0), sleep=lambda s: None,
                                 max_retries=2)


@pytest.mark.parametrize("exc", [ApiError(429, per_day=True), ApiError(429, retry="3600s")])
def test_daily_quota_fails_fast(exc):
    def batch(texts):
        raise exc

    with pytest.raises(inference.EmbeddingQuotaExhausted):
        inference.get_embeddings(["a"], batch_fn=batch, pacer=inference.ItemPacer(0), sleep=pytest.fail)


def test_client_errors_are_not_retried():
    calls = []

    def batch(texts):
        calls.append(1)
        raise ApiError(400)

    with pytest.raises(inference.InferenceUnavailable, match="client_400"):
        inference.get_embeddings(["a"], batch_fn=batch, pacer=inference.ItemPacer(0), sleep=pytest.fail)
    assert len(calls) == 1


def test_pacer_waits_when_the_minute_budget_is_used():
    now, sleeps = [0.0], []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    pacer = inference.ItemPacer(10, clock=lambda: now[0], sleep=sleep)
    pacer.acquire(6)
    pacer.acquire(4)
    pacer.acquire(1)
    assert sleeps == [60.0]


def test_gemini_batch_sends_one_content_per_text(monkeypatch):
    sent = {}

    class Models:
        def embed_content(self, model, contents, config):
            sent.update(model=model, n=len(contents), dim=config.output_dimensionality)
            return type("R", (), {"embeddings": [type("E", (), {"values": [0.0] * 768})() for _ in contents]})()

    monkeypatch.setattr(inference, "_gemini_client", lambda: type("C", (), {"models": Models()})())
    assert len(inference._gemini_batch(["a", "b", "c"])) == 3
    assert sent == {"model": "gemini-embedding-2", "n": 3, "dim": 768}


def test_gemini_batch_count_mismatch_is_an_error(monkeypatch):
    class Models:
        def embed_content(self, model, contents, config):  # the list[str] pitfall: one merged embedding
            return type("R", (), {"embeddings": [type("E", (), {"values": [0.0] * 768})()]})()

    monkeypatch.setattr(inference, "_gemini_client", lambda: type("C", (), {"models": Models()})())
    with pytest.raises(inference.InferenceUnavailable):
        inference._gemini_batch(["a", "b"])


def test_ollama_document_embedding(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5436/documind_local")
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"embeddings": [[0.25] * 768, [0.5] * 768]})

    vecs = inference._ollama_batch(["c1", "c2"], transport=httpx.MockTransport(handler))
    assert len(vecs) == 2 and seen["input"] == ["search_document: c1", "search_document: c2"]
    assert seen["model"] == "nomic-embed-text"


@pytest.mark.parametrize("host", ["localhost", "host.docker.internal", "db"])
def test_ollama_accepts_local_and_compose_database_hosts(monkeypatch, host):
    monkeypatch.setenv("DATABASE_URL", f"postgresql://u:p@{host}:5432/postgres")
    out = inference._ollama_batch(["c"], transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"embeddings": [[0.0] * 768]})))
    assert len(out) == 1


def test_ollama_refuses_shared_database(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@aws-0-eu.pooler.supabase.com:6543/postgres")
    with pytest.raises(inference.InferenceConfigError):
        inference._ollama_batch(["chunk"], transport=httpx.MockTransport(lambda r: httpx.Response(200)))


def test_ollama_unavailable_does_not_leak_text(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:5436/documind_local")

    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(inference.InferenceUnavailable) as info:
        inference._ollama_batch(["secret chunk"], transport=httpx.MockTransport(handler))
    assert "secret" not in str(info.value)
