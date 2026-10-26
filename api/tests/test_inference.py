"""Provider boundary (core/inference.py). No network: Ollama is an httpx.MockTransport."""

import asyncio
import json

import httpx
import pytest

import core.inference as inf


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("INFERENCE_PROVIDER", "OLLAMA_BASE_URL", "OLLAMA_CHAT_MODEL", "OLLAMA_EMBEDDING_MODEL",
                 "OLLAMA_THINK", "OLLAMA_EMBEDDING_PREFIXES", "OLLAMA_NUM_CTX"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen3:14b")  # tests use the lab model name
    inf.reset_provider()
    yield
    inf.reset_provider()


def ollama(handler):
    return inf.OllamaProvider(inf.ollama_settings(), transport=httpx.MockTransport(handler))


# --- selection & configuration ------------------------------------------------------------
def test_gemini_is_the_default_provider(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.example.supabase.co:5432/postgres")
    provider = inf.get_provider()
    assert provider.name == "gemini"
    assert provider.chat_model == "gemini-3.6-flash" and provider.embedding_model == "gemini-embedding-2"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("INFERENCE_PROVIDER", "openai")
    with pytest.raises(inf.InferenceConfigError):
        inf.get_provider()


def test_ollama_defaults(monkeypatch):
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:5436/documind_local")
    provider = inf.get_provider()
    assert provider.name == "ollama"
    assert (provider.chat_model, provider.embedding_model) == ("qwen3:14b", "nomic-embed-text")
    assert provider.settings.base_url == "http://127.0.0.1:11434" and provider.settings.think is False


@pytest.mark.parametrize("env_file,model,base_url", [
    ("env.example", "qwen3:14b", "http://127.0.0.1:11435"),       # lab server via SSH tunnel
    ("env.mac.example", "qwen3:8b", "http://127.0.0.1:11434"),    # Mac's own Ollama
])
def test_environment_templates_select_their_model(monkeypatch, env_file, model, base_url):
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "local_inference" / env_file
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            key, value = line.split("=", 1)
            monkeypatch.setenv(key.strip(), value.strip())
    provider = inf.get_provider()
    assert provider.name == "ollama" and provider.chat_model == model
    assert provider.embedding_model == "nomic-embed-text" and provider.settings.base_url == base_url
    assert provider.settings.embedding_prefixes is True


def test_chat_model_is_required(monkeypatch):
    monkeypatch.delenv("OLLAMA_CHAT_MODEL")
    with pytest.raises(inf.InferenceConfigError, match="OLLAMA_CHAT_MODEL is required"):
        inf.ollama_settings()


@pytest.mark.parametrize("url", [
    "postgresql://postgres.ref:pw@aws-0-eu.pooler.supabase.com:6543/postgres",
    "postgresql://u:p@10.0.0.5:5432/documind",
    "",
])
def test_ollama_refuses_a_non_local_database(monkeypatch, url):
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv("DATABASE_URL", url)
    with pytest.raises(inf.InferenceConfigError, match="hosted database"):
        inf.get_provider()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "host.docker.internal", "db"])
def test_local_database_hosts_are_accepted(monkeypatch, host):
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    monkeypatch.setenv("DATABASE_URL", f"postgresql://u:p@{host}:5432/postgres")
    assert inf.get_provider().name == "ollama"


def test_embedding_space_identifies_provider_model_and_prefixing(monkeypatch):
    monkeypatch.delenv("INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    assert inf.embedding_space() == "gemini:gemini-embedding-2"
    monkeypatch.setenv("INFERENCE_PROVIDER", "ollama")
    assert inf.embedding_space() == "ollama:nomic-embed-text"
    monkeypatch.setenv("OLLAMA_EMBEDDING_PREFIXES", "false")       # different vectors → a different space
    assert inf.embedding_space() == "ollama:nomic-embed-text+noprefix"


def test_api_and_worker_agree_on_the_embedding_space(monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "worker_inference", Path(__file__).resolve().parents[2] / "worker" / "inference.py")
    worker_inf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker_inf)
    for env in ({}, {"INFERENCE_PROVIDER": "ollama"}, {"INFERENCE_PROVIDER": "ollama", "OLLAMA_EMBEDDING_MODEL": "m"},
                {"INFERENCE_PROVIDER": "ollama", "OLLAMA_EMBEDDING_PREFIXES": "0"}, {"EMBEDDING_MODEL": "x"}):
        for key in ("INFERENCE_PROVIDER", "OLLAMA_EMBEDDING_MODEL", "OLLAMA_EMBEDDING_PREFIXES", "EMBEDDING_MODEL"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        assert inf.embedding_space() == worker_inf.embedding_space(), env


def test_invalid_ollama_settings(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "127.0.0.1:11434")
    with pytest.raises(inf.InferenceConfigError):
        inf.ollama_settings()
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_THINK", "maybe")
    with pytest.raises(inf.InferenceConfigError):
        inf.ollama_settings()


# --- embeddings ---------------------------------------------------------------------------
def test_ollama_embedding_uses_task_prefixes_and_checks_dimension():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"embeddings": [[0.5] * 768]})

    provider = ollama(handler)
    assert len(provider.embed("what is ACT?", kind="query")) == 768
    provider.embed("chunk text", kind="document")
    assert seen[0]["input"] == "search_query: what is ACT?"
    assert seen[1]["input"] == "search_document: chunk text"
    assert seen[0]["model"] == "nomic-embed-text"


def test_ollama_embedding_wrong_dimension_is_an_error():
    provider = ollama(lambda r: httpx.Response(200, json={"embeddings": [[0.1] * 1024]}))
    with pytest.raises(inf.InferenceConfigError, match="1024 dims"):
        provider.embed("x", kind="query")


def test_ollama_unreachable_raises_without_fallback():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(inf.InferenceUnavailable, match="ConnectError"):
        ollama(handler).embed("secret document text", kind="document")


def test_ollama_error_does_not_leak_request_text():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(inf.InferenceUnavailable) as info:
        ollama(handler).embed("CONFIDENTIAL PARAGRAPH", kind="document")
    assert "CONFIDENTIAL" not in str(info.value)


def test_ollama_model_missing_is_reported():
    provider = ollama(lambda r: httpx.Response(404, json={"error": "model 'qwen3:14b' not found"}))
    with pytest.raises(inf.InferenceUnavailable, match="HTTP 404.*not found"):
        provider.embed("x")


# --- generation ---------------------------------------------------------------------------
def _ndjson(*events):
    return "\n".join(json.dumps(e) for e in events) + "\n"


def _collect(provider, *args):
    async def run():
        return [t async for t in provider.stream_chat(*args)]

    return asyncio.run(run())


def test_ollama_stream_chat_builds_messages_and_streams_text():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, text=_ndjson(
            {"message": {"role": "assistant", "content": "Hel"}, "done": False},
            {"message": {"role": "assistant", "content": "lo"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True, "eval_count": 2}))

    out = _collect(ollama(handler), "SYSTEM", [("user", "q1"), ("assistant", "a1")], "q2")
    assert out == ["Hel", "lo"]
    assert seen["model"] == "qwen3:14b" and seen["stream"] is True and seen["think"] is False
    assert seen["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "q1"},
                                {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}]
    assert seen["options"]["num_ctx"] == 8192


def test_ollama_stream_error_event_raises():
    handler = lambda r: httpx.Response(200, text=_ndjson({"error": "out of memory"}))  # noqa: E731
    with pytest.raises(inf.InferenceUnavailable, match="out of memory"):
        _collect(ollama(handler), "s", [], "q")


def test_ollama_chat_unreachable_raises():
    def handler(request):
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(inf.InferenceUnavailable, match="ConnectTimeout"):
        _collect(ollama(handler), "s", [], "q")


# --- Gemini path unchanged ----------------------------------------------------------------
def test_gemini_provider_makes_the_original_calls(monkeypatch):
    from types import SimpleNamespace

    calls = {}

    class Models:
        def embed_content(self, *, model, contents, config):
            calls["embed"] = (model, contents, config.output_dimensionality, config.task_type)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.0] * 768)])

    class Stream:
        def __init__(self):
            self.items = [SimpleNamespace(text="a"), SimpleNamespace(text=None), SimpleNamespace(text="b")]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0)

    class AioModels:
        async def generate_content_stream(self, *, model, contents, config):
            calls["chat"] = (model, [(c.role, c.parts[0].text) for c in contents], config.system_instruction)
            return Stream()

    provider = inf.GeminiProvider()
    provider._client = SimpleNamespace(models=Models(), aio=SimpleNamespace(models=AioModels()))
    provider.embed("query", kind="query")
    assert calls["embed"] == ("gemini-embedding-2", "query", 768, None)  # no prefix, no task type
    assert _collect(provider, "SYS", [("user", "q1"), ("assistant", "a1")], "q2") == ["a", "b"]
    assert calls["chat"] == ("gemini-3.6-flash", [("user", "q1"), ("model", "a1"), ("user", "q2")], "SYS")
