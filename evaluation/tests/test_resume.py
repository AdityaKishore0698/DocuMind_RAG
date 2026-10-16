"""Daily-quota stopping, resumable runs, duplicate prevention and partial-answer recovery.

Everything is faked: no network, no API key, no database."""

import json
from types import SimpleNamespace

import pytest

import db
import gemini
import run_eval
from gemini import CallStats, DailyQuotaExhausted, Embedder, Pacer, stream_generate

QUESTIONS = [
    {"id": f"q{i}", "type": "method", "question": f"question {i}?", "status": "approved",
     "gold_answer": "g", "evidence": []}
    for i in range(1, 5)
]


# --- fakes ------------------------------------------------------------------------------
def daily_quota_error():
    from google.genai import errors

    return errors.ClientError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": "You exceeded your current quota. * Quota exceeded for metric: "
                   "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, "
                   "model: gemini-3.6-flash Please retry in 16h30m39.292871476s.",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "59439s"},
        ]}})


def minute_quota_error():
    from google.genai import errors

    return errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                              "message": "Rate limited. Please retry in 25.2s."}})


def server_error():
    from google.genai import errors

    return errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE",
                                              "message": "This model is currently experiencing high demand."}})


class Stream:
    """Yields text chunks; an Exception item is raised mid-stream (after earlier text)."""

    def __init__(self, items):
        self.items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item, usage_metadata=SimpleNamespace(
            prompt_token_count=10, candidates_token_count=2, thoughts_token_count=None,
            cached_content_token_count=None, total_token_count=12))


class FakeClient:
    """outcomes: question text -> list of per-call outcomes (Exception, or list of stream items)."""

    def __init__(self, outcomes):
        self.outcomes = {k: list(v) for k, v in outcomes.items()}
        self.generate_calls: list[str] = []
        self.embed_calls: list[str] = []
        client = self

        class Models:
            def embed_content(self, *, model, contents, config):
                client.embed_calls.append(contents)
                return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3, 0.4])])

        class AioModels:
            async def generate_content_stream(self, *, model, contents, config):
                question = contents[0].parts[0].text
                client.generate_calls.append(question)
                outcome = client.outcomes[question].pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return Stream(outcome)

        self.models = Models()
        self.aio = SimpleNamespace(models=AioModels())


class FakeConn:
    def cursor(self):
        conn = self

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params=None):
                pass

            def fetchall(self):
                return [(p.id, f"{p.id}.pdf") for p in conn.papers]  # (paper_id | doc id, filename)

        return Cursor()

    papers = []


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """run_eval wired to fakes; returns a function that runs `run` with given CLI flags and client."""
    monkeypatch.setenv("EVAL_GENERATE_RPM", "0")
    monkeypatch.setenv("EVAL_EMBED_ITEMS_PER_MIN", "0")
    monkeypatch.setenv("EVAL_MAX_RETRIES", "2")

    async def no_async_sleep(_):
        return None

    monkeypatch.setattr(gemini.asyncio, "sleep", no_async_sleep)
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    monkeypatch.setattr(gemini, "_backoff_seconds", lambda attempt: 0.0)
    monkeypatch.setattr(run_eval, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "load_questions", lambda: [dict(q) for q in QUESTIONS])
    papers = [SimpleNamespace(id="paper1")]
    monkeypatch.setattr(run_eval, "load_corpus", lambda: papers)
    FakeConn.papers = papers
    monkeypatch.setattr(run_eval, "run_config", lambda system, args: {
        "chat_model": system.chat_model, "top_k": system.top_k, "embedding_model": system.embedding_model})
    monkeypatch.setattr(run_eval, "environment_metadata", lambda: {})
    monkeypatch.setattr(run_eval, "_model_metadata", lambda client, models: {})
    monkeypatch.setattr(db, "connect", lambda *a, **k: FakeConn())
    monkeypatch.setattr(db, "corpus_stats", lambda conn: {"chunks": 1003, "documents": 10})
    monkeypatch.setattr(db, "ensure_eval_user", lambda conn: 1)
    monkeypatch.setattr(db, "systems_present", lambda conn: set())
    monkeypatch.setattr(db, "retrieve", lambda conn, sql, user_id, vec, k: [
        {"chunk_id": 1, "paper_id": "paper1", "text": "context", "pages": [1], "cosine_distance": 0.1}])

    def run(client, *flags):
        monkeypatch.setattr(gemini, "make_client", lambda: client)
        args = SimpleNamespace(system="A", questions=None, limit=None, include_drafts=False,
                               fresh_query_embeddings=True, run_id=None, resume=None)
        for i in range(0, len(flags), 2):
            setattr(args, flags[i], flags[i + 1])
        return run_eval.cmd_run(args)

    return run


def _run_dir(tmp_path):
    (only,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    return only


def _lines(run_dir):
    return (run_dir / "traces.jsonl").read_text().splitlines()


# --- daily quota parsing / no retry -------------------------------------------------------
@pytest.mark.parametrize("text,seconds", [("25s", 25.0), ("25.2s", 25.2), ("1m5s", 65.0),
                                          ("16h30m39.29s", 59439.29), ("2h", 7200.0), ("59439s", 59439.0)])
def test_parse_duration(text, seconds):
    assert gemini.parse_duration(text) == pytest.approx(seconds)


def test_parse_duration_rejects_garbage():
    assert gemini.parse_duration("") is None and gemini.parse_duration("soon") is None


def test_daily_quota_is_classified_and_not_retryable():
    assert gemini.classify_error(daily_quota_error()) == "daily_quota"
    assert gemini.classify_error(minute_quota_error()) == "rate_limit"  # per-minute limit is still retried
    # message-only daily limit (no QuotaFailure details): the long delay alone identifies it
    from google.genai import errors

    msg_only = errors.ClientError(429, {"error": {"code": 429, "message": "Please retry in 16h30m39.29s."}})
    assert gemini.retry_after_seconds(msg_only) == pytest.approx(59439.29)
    assert gemini.classify_error(msg_only) == "daily_quota"


def test_stream_generate_does_not_retry_daily_quota():
    stats = CallStats()
    client = FakeClient({"q": [daily_quota_error(), [["never reached"]]]})
    contents = [SimpleNamespace(parts=[SimpleNamespace(text="q")])]
    out = asyncio_run(stream_generate(client, model="m", contents=contents, system_instruction="s",
                                      stats=stats, pacer=Pacer(0, stats), max_retries=4))
    assert len(client.generate_calls) == 1 and stats.retries == 0
    assert out["error"]["daily_quota"] is True and out["error"]["category"] == "daily_quota"
    assert out["error"]["http_status"] == 429 and out["error"]["retry_after_s"] == pytest.approx(59439.0)


def test_embedder_raises_daily_quota_after_one_attempt():
    class Models:
        calls = 0

        def embed_content(self, **kw):
            Models.calls += 1
            raise daily_quota_error()

    emb = Embedder(SimpleNamespace(models=Models()), model="m", dim=4, stats=CallStats(), conn=None, max_retries=4)
    with pytest.raises(DailyQuotaExhausted):
        emb.embed_query("q")
    assert Models.calls == 1 and emb.stats.retries == 0


def test_partial_stream_is_marked_and_not_retried():
    stats = CallStats()
    client = FakeClient({"q": [["Partial ", server_error()]]})
    contents = [SimpleNamespace(parts=[SimpleNamespace(text="q")])]
    out = asyncio_run(stream_generate(client, model="m", contents=contents, system_instruction="s",
                                      stats=stats, pacer=Pacer(0, stats), max_retries=4))
    assert out["error"]["partial_answer"] is True and out["error"]["retry_budget_exhausted"] is False
    assert len(client.generate_calls) == 1
    assert run_eval.trace_outcome({"error": out["error"], "answer": out["text"], "retrieved": []}) == "incomplete"


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


# --- resumable runs --------------------------------------------------------------------------
def test_stop_on_daily_quota_then_resume_to_completion(harness, tmp_path):
    session1 = FakeClient({
        "question 1?": [["Answer one."]],
        "question 2?": [["Partial ", server_error()]],      # partial streamed answer
        "question 3?": [daily_quota_error()],               # daily quota -> stop the session
        "question 4?": [],
    })
    assert harness(session1) == 2
    assert session1.generate_calls == ["question 1?", "question 2?", "question 3?"]  # no retry of q3, q4 untouched
    run_dir = _run_dir(tmp_path)
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["status"] == "stopped_daily_quota"
    assert meta["stop_reason"]["category"] == "daily_quota" and meta["stop_reason"]["http_status"] == 429
    assert meta["outcomes"] == {"q1": "answered", "q2": "incomplete", "q3": "failed_retryable"}
    assert meta["remaining"] == ["q2", "q3", "q4"] and meta["completed"] == ["q1"]
    first_lines = _lines(run_dir)
    assert len(first_lines) == 3
    q3 = json.loads(first_lines[2])
    assert q3["attempt_errors"][0]["http_status"] == 429 and q3["started_at"] and q3["finished_at"]

    session2 = FakeClient({"question 2?": [["Full answer two."]], "question 3?": [["Answer three."]],
                           "question 4?": [["Answer four."]]})
    assert harness(session2, "resume", run_dir.name) == 0
    assert session2.generate_calls == ["question 2?", "question 3?", "question 4?"]  # q1 never regenerated
    lines = _lines(run_dir)
    assert lines[:3] == first_lines  # earlier records preserved byte-for-byte (append-only)
    latest = run_eval.latest_traces([json.loads(x) for x in lines])
    assert latest["q1"]["answer"] == "Answer one."
    assert latest["q2"]["answer"] == "Full answer two."  # regenerated from scratch, not appended to the partial
    assert latest["q2"]["attempt_seq"] == 2 and latest["q2"]["session"] == 2
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["status"] == "completed" and meta["remaining"] == []
    assert [s["session"] for s in meta["sessions"]] == [1, 2]
    assert meta["stats"]["requests"]["generate"] == 1 + 3      # merged across sessions
    assert meta["stats"]["final_failures"] == {"generate:server": 1, "generate:daily_quota": 1}

    # a further resume is a no-op: no calls, no new records
    session3 = FakeClient({})
    assert harness(session3, "resume", run_dir.name) == 0
    assert session3.generate_calls == [] and len(_lines(run_dir)) == 6


def test_report_scores_latest_attempt_once_per_question(harness, tmp_path, monkeypatch):
    harness(FakeClient({"question 1?": [["A1"]], "question 2?": [["Part", server_error()]],
                        "question 3?": [["A3"]], "question 4?": [["A4"]]}))
    run_dir = _run_dir(tmp_path)
    assert json.loads((run_dir / "run.json").read_text())["status"] == "pending_retry"  # q2 incomplete
    harness(FakeClient({"question 2?": [["A2 full"]]}), "resume", run_dir.name)
    records = run_eval.read_traces(run_dir / "traces.jsonl")
    latest = run_eval.latest_traces(records)
    assert len(records) == 5 and len(latest) == 4
    assert [q for q, t in latest.items() if run_eval.trace_outcome(t) == "answered"] == ["q1", "q2", "q3", "q4"]


def test_resume_refuses_changed_configuration(harness, tmp_path, monkeypatch):
    harness(FakeClient({"question 1?": [daily_quota_error()]}))
    run_dir = _run_dir(tmp_path)
    monkeypatch.setattr(run_eval, "run_config", lambda system, args: {"chat_model": "some-other-model",
                                                                      "top_k": system.top_k})
    client = FakeClient({})
    assert harness(client, "resume", run_dir.name) == 1
    assert client.generate_calls == [] and len(_lines(run_dir)) == 1


def test_resume_refuses_changed_corpus(harness, tmp_path, monkeypatch):
    harness(FakeClient({"question 1?": [daily_quota_error()]}))
    monkeypatch.setattr(db, "corpus_stats", lambda conn: {"chunks": 911, "documents": 9})
    assert harness(FakeClient({}), "resume", _run_dir(tmp_path).name) == 1


def test_concurrent_resume_is_refused_by_lock(harness, tmp_path):
    harness(FakeClient({"question 1?": [daily_quota_error()]}))
    run_dir = _run_dir(tmp_path)
    (run_dir / ".lock").write_text("")
    client = FakeClient({})
    assert harness(client, "resume", run_dir.name) == 1 and client.generate_calls == []


def test_answer_loop_refuses_to_duplicate_an_answered_question(harness, tmp_path):
    import asyncio

    prior = [{"question_id": "q1", "outcome": "answered", "error": None, "retrieved": []}]
    args = SimpleNamespace(fresh_query_embeddings=True)
    with pytest.raises(RuntimeError, match="duplicate"):
        asyncio.run(run_eval._answer_questions(run_eval.SYSTEMS["A"], [QUESTIONS[0]], FakeConn(), FakeClient({}),
                                               tmp_path, CallStats(), args, session=2, prior_records=prior))


def test_resume_of_legacy_run_json(harness, tmp_path):
    """The real stopped run predates sessions/outcomes: derive state from its records."""
    run_dir = tmp_path / "legacy_A"
    run_dir.mkdir()
    legacy_records = [
        {"question_id": "q1", "type": "method", "question": "question 1?", "error": None, "retrieved": [],
         "answer": "Answer one.", "timing": {}, "usage": {}, "generation_attempts": 1},
        {"question_id": "q2", "type": "method", "question": "question 2?", "retrieved": [], "answer": "Partial",
         "error": {"category": "server", "http_status": 503, "retry_budget_exhausted": False}},
        {"question_id": "q3", "type": "method", "question": "question 3?", "retrieved": [], "answer": "",
         "error": {"category": "rate_limit", "http_status": 429, "retry_budget_exhausted": True}},
    ]
    (run_dir / "traces.jsonl").write_text("".join(json.dumps(r) + "\n" for r in legacy_records))
    system = run_eval.SYSTEMS["A"]
    legacy_meta = {"run_id": "legacy_A", "system": "A", "description": "d",
                   "config": {"chat_model": system.chat_model, "top_k": system.top_k},
                   "questions_file_sha256": run_eval._sha256(run_eval.QUESTIONS_FILE),
                   "question_ids": ["q1", "q2", "q3", "q4"], "included_drafts": False,
                   "db": {"chunks": 1003}, "environment": {}, "models": {}, "started_at": "t",
                   "status": "stopped_retry_budget_exhausted", "stop_reason": {"category": "rate_limit"},
                   "completed": ["q1", "q2", "q3"], "remaining": ["q4"], "elapsed_s": 5.0,
                   "stats": {"requests": {"generate": 2}, "attempts": {"generate": 7}, "retries": 4, "errors": [],
                             "tokens": {"total_token_count": 100}}}
    (run_dir / "run.json").write_text(json.dumps(legacy_meta))
    client = FakeClient({"question 2?": [["A2"]], "question 3?": [["A3"]], "question 4?": [["A4"]]})
    assert harness(client, "resume", "legacy_A") == 0
    assert client.generate_calls == ["question 2?", "question 3?", "question 4?"]
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["sessions"][0]["status"] == "stopped_retry_budget_exhausted"  # legacy session kept
    assert meta["stats"]["requests"]["generate"] == 2 + 3 and meta["stats"]["tokens"]["total_token_count"] == 100 + 36
    assert meta["config"]["embedding_model"] == system.embedding_model  # missing key recorded, not a mismatch
