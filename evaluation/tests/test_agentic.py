"""Phase 3 graph (agentic.py): routing, bounds, fallbacks, citations — all with fake dependencies."""

import asyncio
import json
from types import SimpleNamespace

import pytest

import agentic
import db
import hybrid
import metrics as M
import ollama_backend
import run_eval
from test_resume import _lines, _run_dir, harness  # noqa: F401  (fixture re-export)


def chunk(cid, paper="octo", text=None):
    return {"chunk_id": cid, "document_id": 100 + cid, "paper_id": paper, "pages": [cid],
            "text": text or f"chunk {cid} text"}


class Fake:
    """Scripted dependencies; records every call."""

    def __init__(self, plan=None, assess=None, rankings=None, answer="Answer [S1]."):
        self.plan, self.assess = plan, list(assess or [])
        self.rankings = rankings or {}
        self.answer = answer
        self.calls, self.prompts = [], []
        self.t = 0.0

    def retrieve(self, query):
        self.calls.append(("retrieve", query))
        return [chunk(c) for c in self.rankings.get(query, [1, 2, 3])]

    def llm_json(self, kind, messages, schema):
        self.calls.append((kind, messages[-1]["content"][:40]))
        if kind == "plan":
            return self.plan, {"parse_error": self.plan is None}
        nxt = self.assess.pop(0) if self.assess else {"sufficient": True, "followup_query": ""}
        return nxt, {"parse_error": nxt is None}

    async def generate(self, system_prompt, question):
        self.calls.append(("synthesize", question))
        self.prompts.append(system_prompt)
        return {"text": self.answer, "ttft_s": 0.5, "total_s": 1.0, "attempts": 1, "error": None, "usage": {}}

    def deps(self):
        return agentic.GraphDeps(retrieve=self.retrieve, llm_json=self.llm_json, generate=self.generate,
                                 source_label=lambda c: f"{c['paper_id']}.pdf", clock=lambda: self.t)


def run(fake, question="Q?"):
    return asyncio.run(agentic.run_graph(fake.deps(), question))


def test_happy_path_single_round():
    fake = Fake(plan={"subqueries": ["octo data", "octo data"]}, rankings={"Q?": [1, 2], "octo data": [2, 3]})
    state = run(fake)
    assert state["queries"] == ["Q?", "octo data"]                      # question first, duplicates removed
    assert [c[0] for c in fake.calls] == ["plan", "retrieve", "retrieve", "assess", "synthesize"]
    assert state["rounds"] == 1 and state["llm_calls"] == {"plan": 1, "assess": 1, "synthesize": 1}
    assert state["context"][0]["chunk_id"] == 2                         # found by both queries → fused first
    assert "[S1] (octo.pdf)" in fake.prompts[0]                          # labelled, numbered sources
    assert state["citations"]["cited_chunk_ids"] == [2]


def test_followup_round_then_stop():
    fake = Fake(plan={"subqueries": []}, assess=[{"sufficient": False, "followup_query": "pi0 dataset"},
                                                  {"sufficient": True, "followup_query": ""}])
    state = run(fake)
    assert state["rounds"] == 2 and ("retrieve", "pi0 dataset") in fake.calls
    assert [a["will_retrieve"] for a in state["assessments"]] == [True, False]


def test_rounds_are_bounded_even_if_never_sufficient():
    fake = Fake(plan={"subqueries": []},
                assess=[{"sufficient": False, "followup_query": f"q{i}"} for i in range(10)])
    state = run(fake)
    assert state["rounds"] == agentic.MAX_ROUNDS
    assert sum(1 for c in fake.calls if c[0] == "assess") == agentic.MAX_ROUNDS
    assert fake.calls[-1][0] == "synthesize"


def test_repeated_followup_ends_loop_and_time_budget_respected():
    fake = Fake(plan={"subqueries": []}, assess=[{"sufficient": False, "followup_query": "Q?"}])
    assert run(fake)["rounds"] == 1                                     # same query again → no new retrieval
    slow = Fake(plan={"subqueries": []}, assess=[{"sufficient": False, "followup_query": "new"}])
    slow.t = 0.0
    deps = slow.deps()
    deps.clock = iter([0.0] + [agentic.TIME_BUDGET_S + 1] * 20).__next__
    state = asyncio.run(agentic.run_graph(deps, "Q?"))
    assert state["rounds"] == 1                                         # over budget → straight to synthesis


def test_unparseable_json_falls_back_deterministically():
    fake = Fake(plan=None, assess=[None])
    state = run(fake)
    assert state["queries"] == ["Q?"] and state["rounds"] == 1
    assert state["parse_errors"] == {"plan": 1, "assess": 1}
    assert fake.calls[-1][0] == "synthesize"


def test_context_budget_is_five_chunks():
    fake = Fake(plan={"subqueries": ["a", "b"]},
                rankings={"Q?": [1, 2, 3, 4, 5], "a": [6, 7, 8, 9, 10], "b": [11, 12, 13, 14, 15]})
    state = run(fake)
    assert len(state["context"]) == agentic.CONTEXT_BUDGET == 5
    assert [c["rank"] for c in state["context"]] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("answer,valid,invalid,markers", [
    ("Octo uses diffusion [S1]. It has 800k [S2][S3].", [1, 2, 3], [], 3),
    ("Both [S1, S2] and [s4].", [1, 2, 4], [], 3),
    ("Made up [S9].", [], [9], 1),
    ("No citations at all.", [], [], 0),
])
def test_parse_citations(answer, valid, invalid, markers):
    out = agentic.parse_citations(answer, n_sources=5)
    assert (out["valid_sources"], out["invalid_sources"], out["markers"]) == (valid, invalid, markers)


def test_verify_flags_uncited_answers_but_not_abstentions():
    assert run(Fake(plan={"subqueries": []}, answer="Octo uses diffusion."))["citations"]["uncited_answer"] is True
    abstain = run(Fake(plan={"subqueries": []}, answer="The sources do not contain this information."))
    assert abstain["citations"]["uncited_answer"] is False and abstain["citations"]["abstained"] is True


def test_score_citations():
    q = {"evidence": [{"paper": "octo", "page": 1, "quote": "800k trajectories"}]}
    ctx = [chunk(1, text="Trained on 800k trajectories."), chunk(2, text="unrelated")]
    out = M.score_citations(q, {"valid_sources": [1, 2], "invalid_sources": [7], "markers": 3}, ctx)
    assert out == {"citation_validity": pytest.approx(2 / 3), "citation_support_precision": 0.5,
                   "answer_has_citations": 1.0}
    assert M.score_citations({"evidence": []}, {"valid_sources": [], "invalid_sources": [], "markers": 0}, ctx) == {
        "citation_validity": None, "citation_support_precision": None, "answer_has_citations": None}


def test_chat_json_counts_all_tokens_and_handles_bad_json():
    import httpx

    from gemini import CallStats

    replies = iter([{"message": {"content": '{"subqueries": ["a"]}'}, "prompt_eval_count": 7, "eval_count": 3},
                    {"message": {"content": "not json"}, "prompt_eval_count": 2, "eval_count": 1}])
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json=next(replies)))
    stats = CallStats()
    parsed, info = ollama_backend.chat_json(model="m", messages=[], schema={}, stats=stats, kind="plan",
                                            transport=transport)
    assert parsed == {"subqueries": ["a"]} and info["usage"]["total_token_count"] == 10
    parsed, info = ollama_backend.chat_json(model="m", messages=[], schema={}, stats=stats, kind="plan",
                                            transport=transport)
    assert parsed is None and info["parse_error"] is True
    assert stats.tokens == {"prompt_token_count": 9, "candidates_token_count": 4, "total_token_count": 13}


def test_controlled_systems_differ_only_where_intended():
    L, BL, CL, CLV = (run_eval.SYSTEMS[n] for n in ("L", "BL", "CL", "CLV"))
    for s in (BL, CL, CLV):
        assert s.corpus == "L" and (s.chat_model, s.embedding_model, s.top_k) == (L.chat_model, L.embedding_model, 5)
    assert (BL.mode, BL.retrieval) == ("oneshot", "hybrid_bm25") and BL.build_system_prompt is L.build_system_prompt
    assert (CL.mode, CL.retrieval, CLV.mode, CLV.retrieval) == ("graph", "hybrid_bm25", "graph", "vector")
    assert run_eval.SYSTEMS["A"].mode == "oneshot" and run_eval.SYSTEMS["A"].retrieval == "vector"


def test_graph_system_through_the_harness(harness, monkeypatch, tmp_path):  # noqa: F811
    class Emb:
        pacer = SimpleNamespace(limit=0)

        def embed_query(self, text):
            return [0.0] * 768, 0.01, False

    def fake_chat_json(*, model, messages, schema, stats, kind, max_retries=2, transport=None, sleep=None):
        stats.bump(stats.requests, kind)
        stats.bump(stats.tokens, "prompt_token_count", 5)
        return ({"subqueries": []} if kind == "plan" else {"sufficient": True, "followup_query": ""}), {}

    async def fake_generate(*, model, system_instruction, question, stats, max_retries=2, transport=None):
        stats.bump(stats.requests, "generate")
        return {"text": "It is X [S1].", "ttft_s": 0.2, "total_s": 0.4, "attempts": 1, "error": None, "usage": {}}

    monkeypatch.setattr(run_eval, "make_embedder", lambda system, **kw: Emb())
    monkeypatch.setattr(ollama_backend, "chat_json", fake_chat_json)
    monkeypatch.setattr(ollama_backend, "stream_generate", fake_generate)
    monkeypatch.setattr(ollama_backend, "model_metadata", lambda models: {"models": {}})
    monkeypatch.setattr(hybrid, "assert_lexical_index_fresh", lambda conn: None)
    monkeypatch.setattr(hybrid, "hybrid_retrieve", lambda conn, **kw: [chunk(1), chunk(2)])
    monkeypatch.setattr(db, "systems_present", lambda conn: {"L"})
    assert harness(None, "system", "CL") == 0
    t = json.loads(_lines(_run_dir(tmp_path))[0])
    assert t["outcome"] == "answered" and t["graph"]["llm_calls"] == {"plan": 1, "assess": 1, "synthesize": 1}
    assert t["graph"]["citations"]["cited_chunk_ids"] == [1]
    assert t["retrieved"][0]["source_label"] == "unknown"  # fake DB has no filename for document 101
    assert t["timing"]["ttft_s"] >= 0.2 and t["usage"]["prompt_token_count"] == 10
