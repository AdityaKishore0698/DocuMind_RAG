"""Gemini layer with fake clients — no network, no API key, no quota."""

import asyncio
from types import SimpleNamespace

import pytest

import gemini
from gemini import CallStats, Embedder, Pacer, classify_error, stream_generate


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


def _vec(seed: float, dim: int = 4):
    return [seed + i for i in range(dim)]


class FakeModels:
    """Mimics models.embed_content: one embedding per Content; a plain str -> one embedding."""

    def __init__(self, fail_first_with=None, collapse_batches=False):
        self.calls = []
        self.fail_first_with = fail_first_with
        self.collapse_batches = collapse_batches

    def embed_content(self, *, model, contents, config):
        self.calls.append(contents)
        if self.fail_first_with is not None and len(self.calls) == 1:
            raise FakeAPIError(self.fail_first_with)
        items = contents if isinstance(contents, list) else [contents]
        if self.collapse_batches and isinstance(contents, list):
            items = items[:1]  # simulate the list[str] -> single aggregated embedding pitfall
        return SimpleNamespace(embeddings=[SimpleNamespace(values=_vec(float(len(self.calls)))) for _ in items])


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(gemini.time, "sleep", lambda s: None)
    monkeypatch.setattr(gemini, "_backoff_seconds", lambda attempt: 0.0)


def make_embedder(models, **kw):
    return Embedder(SimpleNamespace(models=models), model="gemini-embedding-2", dim=4, stats=CallStats(),
                    conn=None, **kw)


def test_batch_sends_one_content_object_per_text():
    from google.genai import types

    models = FakeModels()
    emb = make_embedder(models, batch_size=10)
    vectors = emb.embed_many(["a", "b", "c"])
    assert len(vectors) == 3 and len(models.calls) == 1
    sent = models.calls[0]
    assert all(isinstance(c, types.Content) and len(c.parts) == 1 for c in sent)
    assert [c.parts[0].text for c in sent] == ["a", "b", "c"]
    assert emb.stats.requests == {"embed_batch": 1}


def test_count_mismatch_falls_back_to_production_single_calls():
    models = FakeModels(collapse_batches=True)
    emb = make_embedder(models, batch_size=10)
    vectors = emb.embed_many(["a", "b", "c"])
    assert len(vectors) == 3
    assert emb.stats.batch_fallbacks == 1
    assert models.calls[1:] == ["a", "b", "c"]  # per-item calls use the exact production shape (a str)


def test_single_mode_reproduces_production_call_shape():
    models = FakeModels()
    emb = make_embedder(models, mode="single")
    emb.embed_many(["x", "y"])
    assert models.calls == ["x", "y"]


def test_batches_respect_batch_size_and_dedupe():
    models = FakeModels()
    emb = make_embedder(models, batch_size=2)
    vectors = emb.embed_many(["a", "b", "c", "a"])
    assert len(models.calls) == 2  # 3 unique texts -> batches of 2 + 1
    assert vectors[0] == vectors[3]


def test_retry_on_429_counts_first_attempt_error_and_retry():
    models = FakeModels(fail_first_with=429)
    emb = make_embedder(models, batch_size=10, max_retries=2)
    emb.embed_many(["a"])
    assert emb.stats.first_attempt_errors == {"embed_batch:rate_limit": 1}
    assert emb.stats.retries == 1 and emb.stats.final_failures == {}


def test_client_errors_are_not_retried():
    models = FakeModels(fail_first_with=400)
    emb = make_embedder(models, batch_size=10, max_retries=3)
    with pytest.raises(FakeAPIError):
        emb.embed_many(["a"])
    assert emb.stats.retries == 0 and emb.stats.final_failures == {"embed_batch:client_400": 1}


def test_classify_error():
    assert classify_error(FakeAPIError(429)) == "rate_limit"
    assert classify_error(FakeAPIError(503)) == "server"
    assert classify_error(FakeAPIError(404)) == "client_404"
    assert classify_error(TimeoutError()) == "timeout"
    assert classify_error(ValueError()) == "other"


# --- streaming generation ---------------------------------------------------------------
class FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _chunk(text, usage=None):
    return SimpleNamespace(text=text, usage_metadata=usage)


class FakeAio:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def generate_content_stream(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "system_instruction": config.system_instruction})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeStream(outcome)


def _client(outcomes):
    return SimpleNamespace(aio=SimpleNamespace(models=FakeAio(outcomes)))


def test_stream_generate_collects_text_ttft_and_usage():
    usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=50,
                            cached_content_token_count=None, total_token_count=170)
    client = _client([[_chunk("Hel"), _chunk("lo", usage)]])
    stats = CallStats()
    out = asyncio.run(stream_generate(client, model="m", contents=["q"], system_instruction="sys",
                                      stats=stats, pacer=Pacer(0, stats)))
    assert out["text"] == "Hello" and out["error"] is None and out["ttft_s"] is not None
    assert out["usage"] == {"prompt_token_count": 100, "candidates_token_count": 20,
                            "thoughts_token_count": 50, "total_token_count": 170}
    assert stats.tokens["total_token_count"] == 170
    assert client.aio.models.calls[0]["system_instruction"] == "sys"


def test_stream_generate_retries_429_then_succeeds(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(gemini.asyncio, "sleep", no_sleep)
    client = _client([FakeAPIError(429), [_chunk("ok")]])
    stats = CallStats()
    out = asyncio.run(stream_generate(client, model="m", contents=["q"], system_instruction="s",
                                      stats=stats, pacer=Pacer(0, stats), max_retries=2))
    assert out["text"] == "ok" and out["attempts"] == 2
    assert stats.first_attempt_errors == {"generate:rate_limit": 1} and stats.retries == 1


def test_stream_generate_records_final_failure_without_raising():
    client = _client([FakeAPIError(400)])
    stats = CallStats()
    out = asyncio.run(stream_generate(client, model="m", contents=["q"], system_instruction="s",
                                      stats=stats, pacer=Pacer(0, stats)))
    assert out["error"]["category"] == "client_400" and out["error"]["type"] == "FakeAPIError"
    assert out["error"]["retry_budget_exhausted"] is False  # a client error is not a quota stop
    assert stats.final_failures == {"generate:client_400": 1}


# --- quota handling ------------------------------------------------------------------------
QUOTA_DETAILS = {"error": {
    "code": 429, "status": "RESOURCE_EXHAUSTED",
    "message": "You exceeded your current quota ... Please retry in 25.212948056s.",
    "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{
            "quotaMetric": "generativelanguage.googleapis.com/embed_content_free_tier_requests",
            "quotaId": "EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier", "quotaValue": "100"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "25s"},
    ]}}


class FakeQuotaError(FakeAPIError):
    def __init__(self, details=QUOTA_DETAILS):
        super().__init__(429)
        self.details = details


def test_retry_after_parsed_from_retryinfo_then_message():
    assert gemini.retry_after_seconds(FakeQuotaError()) == 25.0
    no_retryinfo = {"error": {**QUOTA_DETAILS["error"], "details": []}}
    assert gemini.retry_after_seconds(FakeQuotaError(no_retryinfo)) == pytest.approx(25.212948056)
    assert gemini.retry_after_seconds(FakeAPIError(429)) is None


def test_quota_info_extracts_violation():
    (v,) = gemini.quota_info(FakeQuotaError())
    assert v["quotaId"].endswith("FreeTier") and v["quotaValue"] == "100"


def test_retry_wait_respects_server_delay_plus_buffer_and_cap(monkeypatch):
    monkeypatch.setattr(gemini, "_backoff_seconds", lambda attempt: 1.0)
    assert gemini.retry_wait_seconds(1, FakeQuotaError()) == 25.0 + gemini.RETRY_BUFFER_S
    assert gemini.retry_wait_seconds(1, FakeAPIError(503)) == 1.0          # no server hint -> back-off
    monkeypatch.setattr(gemini, "_backoff_seconds", lambda attempt: 999.0)
    assert gemini.retry_wait_seconds(5, FakeAPIError(503)) == gemini.MAX_SINGLE_WAIT_S


class AlwaysQuota(FakeModels):
    def embed_content(self, *, model, contents, config):
        self.calls.append(contents)
        raise FakeQuotaError()


def test_retry_budget_exhausted_stops_and_records_everything():
    slept = []
    models = AlwaysQuota()
    emb = Embedder(SimpleNamespace(models=models), model="gemini-embedding-2", dim=4, stats=CallStats(),
                   conn=None, batch_size=3, max_retries=2, sleep=slept.append)
    with pytest.raises(gemini.RetryBudgetExhausted) as info:
        emb.embed_many(["a", "b", "c"])
    s = emb.stats
    assert len(models.calls) == 3 and s.attempts == {"embed_batch": 3}  # 1 + 2 retries, never more
    assert s.retries == 2 and s.embed_items_attempted == 9 and s.embed_items_embedded == 0
    assert s.final_failures == {"embed_batch:rate_limit": 1}
    assert all(w >= 25.0 + gemini.RETRY_BUFFER_S for w in slept)        # honoured the server delay
    assert info.value.quota[0]["quotaValue"] == "100" and info.value.retry_after_s == 25.0
    assert len(s.errors) == 3 and s.errors[-1]["category"] == "rate_limit"


def test_item_pacer_counts_items_in_a_sliding_window():
    now = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    pacer = gemini.ItemPacer(90, CallStats(), clock=lambda: now[0], sleep=sleep)
    pacer.acquire(45)
    pacer.acquire(45)            # 90 items within the window: allowed without waiting
    assert slept == []
    now[0] += 10
    pacer.acquire(45)            # would be 135 > 90 -> wait until the first batch leaves the window
    assert slept and now[0] >= 60.0
    assert pacer.used() <= 90
    with pytest.raises(ValueError):
        pacer.acquire(91)        # a single batch larger than the budget is rejected


def test_batch_size_is_clamped_to_item_budget():
    emb = Embedder(SimpleNamespace(models=FakeModels()), model="m", dim=4, stats=CallStats(),
                   batch_size=100, items_per_min=90)
    assert emb.batch_size == 90


def test_stream_generate_marks_exhausted_retry_budget(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(gemini.asyncio, "sleep", no_sleep)
    client = _client([FakeQuotaError(), FakeQuotaError()])
    stats = CallStats()
    out = asyncio.run(stream_generate(client, model="m", contents=["q"], system_instruction="s",
                                      stats=stats, pacer=Pacer(0, stats), max_retries=1))
    assert out["error"]["retry_budget_exhausted"] is True and out["error"]["retry_after_s"] == 25.0
    assert stats.attempts == {"generate": 2} and stats.retries == 1


# --- sanitized error details --------------------------------------------------------------
def _server_error(message="The model is overloaded. Please try again later."):
    from google.genai import errors

    return errors.ServerError(503, {"error": {"code": 503, "message": message, "status": "UNAVAILABLE"}})


def test_error_details_from_real_sdk_server_error():
    d = gemini.error_details(_server_error())
    assert d == {"type": "ServerError", "http_status": 503, "status": "UNAVAILABLE",
                 "message": "The model is overloaded. Please try again later."}
    assert classify_error(_server_error()) == "server"


def test_error_details_redacts_credentials_and_bounds_length():
    fake_key = "AIza" + "x" * 35
    d = gemini.error_details(_server_error(
        f"bad request ?key={fake_key} Bearer abc.def-ghi raw {fake_key} " + "y" * 2000))
    assert fake_key not in d["message"] and "abc.def-ghi" not in d["message"]
    assert "[REDACTED]" in d["message"]
    assert len(d["message"]) <= gemini.MAX_ERROR_MESSAGE + 1


def test_generation_failure_records_status_and_message_but_not_the_prompt(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(gemini.asyncio, "sleep", no_sleep)
    secret_prompt = "PAPER CONTEXT THAT MUST NOT BE LOGGED"
    stats = CallStats()
    client = _client([_server_error(), _server_error()])
    out = asyncio.run(stream_generate(client, model="m", contents=secret_prompt, system_instruction=secret_prompt,
                                      stats=stats, pacer=Pacer(0, stats), max_retries=1))
    err = out["error"]
    assert err["http_status"] == 503 and err["status"] == "UNAVAILABLE" and err["type"] == "ServerError"
    assert err["retry_budget_exhausted"] is True
    assert [e["attempt"] for e in stats.errors] == [1, 2]
    assert all(e["http_status"] == 503 and e["at"] for e in stats.errors)
    assert stats.errors[0]["waited_s"] is not None
    assert secret_prompt not in repr(stats.as_dict()) + repr(out)
