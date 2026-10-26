"""System L (local-model baseline via Ollama): backend, configuration, isolation, incremental runs.

No network: Ollama is an httpx.MockTransport or a fake; no database."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

import baseline_a
import db
import ollama_backend
import run_eval
from gemini import CallStats
from test_resume import FakeConn, QUESTIONS, _lines, _run_dir, harness  # noqa: F401  (fixture re-export)


# --- System L configuration ------------------------------------------------------------------
def test_system_l_reuses_system_a_pipeline_and_differs_only_in_models():
    a, l = run_eval.SYSTEMS["A"], run_eval.SYSTEMS["L"]
    assert (l.provider, l.chat_model, l.embedding_model, l.embedding_dim) == ("ollama", "qwen3:8b",
                                                                                "nomic-embed-text", 768)
    for field in ("top_k", "retrieval_sql", "extract_text", "split_text", "build_system_prompt", "build_context",
                  "preprocess"):
        assert getattr(l, field) is getattr(a, field) or getattr(l, field) == getattr(a, field), field
    assert a.provider == "gemini" and a.chat_model == baseline_a.CHAT_MODEL  # Baseline A untouched
    assert "NOT the Gemini Baseline A" in l.description


def test_run_config_records_provider_and_models():
    args = SimpleNamespace(fresh_query_embeddings=False)
    cfg = run_eval.run_config(run_eval.SYSTEMS["L"], args)
    assert cfg["provider"] == "ollama" and cfg["chat_model"] == "qwen3:8b"
    assert cfg["embedding_model"] == "nomic-embed-text"
    assert any("nomic-embed-text" in d for d in cfg["deviations"])


# --- embedding-space isolation ------------------------------------------------------------------
def test_one_embedding_space_per_database(monkeypatch):
    monkeypatch.setattr(db, "systems_present", lambda conn: {"A"})
    with pytest.raises(SystemExit, match="refusing to mix embedding spaces"):
        run_eval.assert_single_embedding_space(None, run_eval.SYSTEMS["L"])
    run_eval.assert_single_embedding_space(None, run_eval.SYSTEMS["A"])  # same system is fine
    monkeypatch.setattr(db, "systems_present", lambda conn: {"L"})
    with pytest.raises(SystemExit):
        run_eval.assert_single_embedding_space(None, run_eval.SYSTEMS["A"])


# --- Ollama backend -----------------------------------------------------------------------------------
def test_embedder_prefixes_batches_and_counts():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"embeddings": [[0.1] * 768 for _ in body["input"]]})

    emb = ollama_backend.OllamaEmbedder(model="nomic-embed-text", dim=768, stats=CallStats(), batch_size=2,
                                        transport=httpx.MockTransport(handler))
    vectors = emb.embed_many(["a", "b", "c", "a"])
    assert len(vectors) == 4 and [len(c["input"]) for c in calls] == [2, 1]  # deduped, batched
    assert calls[0]["input"] == ["search_document: a", "search_document: b"]
    vec, _, cached = emb.embed_query("q?")
    assert calls[-1]["input"] == ["search_query: q?"] and not cached and len(vec) == 768
    assert emb.stats.requests == {"embed_batch": 2, "embed_query": 1}


def test_embedder_rejects_wrong_dimension_and_unavailable_server():
    bad = ollama_backend.OllamaEmbedder(model="m", dim=768, stats=CallStats(), transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"embeddings": [[0.1] * 1024]})))
    with pytest.raises(RuntimeError, match="768"):
        bad.embed_many(["x"])

    def down(request):
        raise httpx.ConnectError("refused")

    stats = CallStats()
    emb = ollama_backend.OllamaEmbedder(model="m", dim=768, stats=stats, max_retries=1,
                                        transport=httpx.MockTransport(down), sleep=lambda s: None)
    with pytest.raises(ollama_backend.OllamaError) as info:
        emb.embed_query("q")
    assert info.value.details["category"] == "unavailable" and stats.retries == 1


def _ndjson(*events):
    return "\n".join(json.dumps(e) for e in events) + "\n"


def test_stream_generate_maps_usage_and_timings():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, text=_ndjson(
            {"message": {"content": "Twelve "}, "done": False},
            {"message": {"content": "hours."}, "done": False},
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 900, "eval_count": 4,
             "load_duration": 2_000_000_000, "prompt_eval_duration": 1_500_000_000, "eval_duration": 300_000_000}))

    stats = CallStats()
    out = asyncio.run(ollama_backend.stream_generate(model="qwen3:8b", system_instruction="SYS", question="Q?",
                                                     stats=stats, transport=httpx.MockTransport(handler)))
    assert out["text"] == "Twelve hours." and out["error"] is None and out["ttft_s"] is not None
    assert out["usage"] == {"prompt_token_count": 900, "candidates_token_count": 4, "total_token_count": 904}
    assert out["server_timings_s"] == {"load_duration": 2.0, "prompt_eval_duration": 1.5, "eval_duration": 0.3}
    assert seen["think"] is False and seen["options"]["num_ctx"] == 8192
    assert seen["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q?"}]


def test_stream_generate_partial_answer_is_incomplete_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, text=_ndjson({"message": {"content": "Part"}, "done": False},
                                                {"error": "model runner crashed"}))

    out = asyncio.run(ollama_backend.stream_generate(model="m", system_instruction="s", question="q",
                                                     stats=CallStats(), transport=httpx.MockTransport(handler)))
    assert out["error"]["partial_answer"] is True and len(calls) == 1
    assert run_eval.trace_outcome({"error": out["error"], "answer": out["text"], "retrieved": []}) == "incomplete"


# --- incremental local run (fakes) ------------------------------------------------------------------
@pytest.fixture
def local_fakes(monkeypatch):
    calls = []

    class Emb:
        pacer = SimpleNamespace(limit=0)

        def embed_query(self, text):
            return [0.0] * 768, 0.01, False

    async def fake_generate(*, model, system_instruction, question, stats, max_retries=2, transport=None):
        calls.append((model, question))
        stats.bump(stats.requests, "generate")
        stats.bump(stats.tokens, "total_token_count", 10)
        return {"text": f"answer to {question}", "ttft_s": 0.1, "total_s": 0.2, "attempts": 1, "error": None,
                "usage": {"total_token_count": 10}, "server_timings_s": {"eval_duration": 0.1}}

    monkeypatch.setattr(run_eval, "make_embedder", lambda system, **kw: Emb())
    monkeypatch.setattr(ollama_backend, "stream_generate", fake_generate)
    monkeypatch.setattr(ollama_backend, "model_metadata", lambda models: {"models": {m: {} for m in models}})
    return calls


def test_local_run_in_batches_then_resume(harness, local_fakes, tmp_path):  # noqa: F811
    assert harness(None, "system", "L", "max_questions", 2) == 2      # paused after a small batch
    run_dir = _run_dir(tmp_path)
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["status"] == "paused" and meta["remaining"] == ["q3", "q4"]
    assert meta["system"] == "L" and "NOT the Gemini Baseline A" in meta["description"]
    assert set(meta["models"]["models"]) == {"qwen3:8b", "nomic-embed-text"}
    assert [m for m, _ in local_fakes] == ["qwen3:8b", "qwen3:8b"]
    first = _lines(run_dir)
    assert harness(None, "resume", run_dir.name) == 0                 # continues the SAME run
    assert [q for _, q in local_fakes] == ["question 1?", "question 2?", "question 3?", "question 4?"]
    assert _lines(run_dir)[:2] == first
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["status"] == "completed" and meta["stats"]["tokens"]["total_token_count"] == 40
    assert json.loads(_lines(run_dir)[0])["timing"]["server_s"] == {"eval_duration": 0.1}
