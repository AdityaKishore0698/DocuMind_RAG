"""Phase 3 — bounded LangGraph workflow for research-question answering (evaluation-only).

    plan ─▶ retrieve ─▶ assess ─(insufficient ∧ round < MAX_ROUNDS ∧ within budget)─▶ retrieve
                           └──────────────(otherwise)──────────────▶ synthesize ─▶ verify ─▶ END

* plan        — one JSON call: ≤ MAX_SUBQUERIES search queries (the original question is always query 1).
                Invalid JSON → deterministic fallback to the question alone.
* retrieve    — each *new* query through the selected method (vector or BM25 hybrid, top PER_QUERY_K);
                per-query rankings fused with the existing RRF (hybrid.rrf_fuse).
* assess      — one JSON call: is the evidence sufficient? if not, one follow-up query. Bounded by
                MAX_ROUNDS retrieval rounds and a wall-clock budget; a repeated query ends the loop.
* synthesize  — streamed answer from the top CONTEXT_BUDGET fused chunks (same 5-chunk budget as the
                one-shot baselines), sources labelled [S1].. with their document filename; every claim cited.
* verify      — deterministic: citation markers parsed, checked against the sources actually given,
                mapped to chunk ids. No LLM judge, no rewriting.

Dependencies (retrieval, JSON calls, streamed generation) are injected, so the graph logic is tested
without a model or database (tests/test_agentic.py).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TypedDict

MAX_SUBQUERIES = 3
MAX_ROUNDS = 2
PER_QUERY_K = 5
CONTEXT_BUDGET = 5
TIME_BUDGET_S = 300.0
EXCERPT_CHARS = 600  # assessment sees truncated excerpts only (keeps the check cheap)

PLAN_SYSTEM = (
    "You plan document retrieval for questions about robot-learning research papers. "
    "Return JSON {\"subqueries\": [...]} with up to 3 short search queries that together cover everything the "
    "question asks (for comparisons, one query per paper or per aspect). Do not answer the question.")
PLAN_SCHEMA = {"type": "object", "properties": {"subqueries": {"type": "array", "items": {"type": "string"}}},
               "required": ["subqueries"]}

ASSESS_SYSTEM = (
    "You check whether retrieved excerpts contain the evidence needed to answer a question about research papers. "
    "Return JSON {\"sufficient\": true|false, \"missing\": \"what is missing\", \"followup_query\": \"a search query "
    "for the missing evidence, or empty\"}. Judge only from the excerpts.")
ASSESS_SCHEMA = {"type": "object", "properties": {"sufficient": {"type": "boolean"}, "missing": {"type": "string"},
                                                  "followup_query": {"type": "string"}},
                 "required": ["sufficient", "followup_query"]}

SYNTH_SYSTEM = (
    "You are DocuMind, a research-paper analyst. Answer the question using only the numbered sources below. "
    "Cite the supporting source after every claim as [S1], [S2], ... Keep facts attributed to the paper they come "
    "from. If the sources do not contain the answer, say so explicitly and do not guess.\n\nSources:\n{sources}")

_CITE_GROUP = re.compile(r"\[([^\[\]]{1,40})\]")
_CITE_ID = re.compile(r"S\s*(\d+)", re.IGNORECASE)
_ABSTAIN = re.compile(r"(do(es)? not (contain|provide|mention|specify)|not (stated|specified|mentioned|provided) in|"
                      r"cannot (be )?(determine|answer|find)|no information)", re.IGNORECASE)


class GraphState(TypedDict, total=False):
    question: str
    started: float
    queries: list[str]                 # all queries issued so far (deduplicated, in order)
    rankings: dict[str, list[int]]     # query -> ranked chunk ids
    chunks: dict[int, dict]            # chunk id -> retrieved chunk
    rounds: int
    assessments: list[dict]
    pending: list[str]                 # queries not yet retrieved
    context: list[dict]
    generation: dict
    citations: dict
    events: list[dict]
    llm_calls: dict[str, int]
    parse_errors: dict[str, int]


@dataclass
class GraphDeps:
    retrieve: Callable[[str], list[dict]]                                   # query -> ranked chunks
    llm_json: Callable[[str, list[dict], dict], tuple[dict | None, dict]]   # (kind, messages, schema)
    generate: Callable[[str, str], Awaitable[dict]]                         # (system prompt, question) -> gen dict
    source_label: Callable[[dict], str]                                     # chunk -> "octo.pdf"
    clock: Callable[[], float] = time.perf_counter


def parse_citations(answer: str, n_sources: int) -> dict:
    """[S1], [S1, S3], [S2][S4] → marker list; valid = refers to a source that was provided."""
    ids = [int(m) for g in _CITE_GROUP.findall(answer or "") for m in _CITE_ID.findall(g)]
    valid = sorted({i for i in ids if 1 <= i <= n_sources})
    invalid = sorted({i for i in ids if not 1 <= i <= n_sources})
    return {"markers": len(ids), "valid_sources": valid, "invalid_sources": invalid,
            "abstained": bool(_ABSTAIN.search(answer or ""))}


def _bump(d: dict | None, key: str) -> dict:
    d = dict(d or {})
    d[key] = d.get(key, 0) + 1
    return d


def build_graph(deps: GraphDeps):
    from langgraph.graph import END, START, StateGraph

    import hybrid

    def plan(state: GraphState) -> dict:
        question = state["question"]
        parsed, info = deps.llm_json("plan", [{"role": "system", "content": PLAN_SYSTEM},
                                              {"role": "user", "content": question}], PLAN_SCHEMA)
        subqueries = []
        if parsed and isinstance(parsed.get("subqueries"), list):
            subqueries = [q.strip() for q in parsed["subqueries"] if isinstance(q, str) and q.strip()]
        queries = list(dict.fromkeys([question] + subqueries[:MAX_SUBQUERIES]))
        return {"queries": queries, "pending": queries, "rounds": 0, "rankings": {}, "chunks": {},
                "assessments": [], "llm_calls": _bump(state.get("llm_calls"), "plan"),
                "parse_errors": _bump(state.get("parse_errors"), "plan") if info.get("parse_error")
                else dict(state.get("parse_errors") or {}),
                "events": (state.get("events") or []) + [{"node": "plan", "queries": queries,
                                                          "parse_error": info.get("parse_error", False)}]}

    def retrieve(state: GraphState) -> dict:
        rankings, chunks = dict(state["rankings"]), dict(state["chunks"])
        for query in state["pending"]:
            results = deps.retrieve(query)[:PER_QUERY_K]
            rankings[query] = [r["chunk_id"] for r in results]
            for r in results:
                chunks.setdefault(r["chunk_id"], r)
        return {"rankings": rankings, "chunks": chunks, "pending": [], "rounds": state["rounds"] + 1,
                "events": state["events"] + [{"node": "retrieve", "round": state["rounds"] + 1,
                                              "queries": list(state["pending"]), "pool": len(chunks)}]}

    def fused(state: GraphState, k: int) -> list[dict]:
        order = hybrid.rrf_fuse(state["rankings"], top_k=k) if state["rankings"] else []
        return [state["chunks"][f["chunk_id"]] for f in order]

    def assess(state: GraphState) -> dict:
        top = fused(state, CONTEXT_BUDGET)
        excerpts = "\n\n".join(f"[{i}] ({deps.source_label(c)}) {c['text'][:EXCERPT_CHARS]}"
                               for i, c in enumerate(top, start=1))
        parsed, info = deps.llm_json("assess", [
            {"role": "system", "content": ASSESS_SYSTEM},
            {"role": "user", "content": f"Question: {state['question']}\n\nExcerpts:\n{excerpts}"}], ASSESS_SCHEMA)
        sufficient = bool(parsed.get("sufficient")) if parsed else True  # unparseable → stop looping
        followup = (parsed.get("followup_query") or "").strip() if parsed else ""
        pending = []
        if (not sufficient and followup and followup not in state["queries"] and state["rounds"] < MAX_ROUNDS
                and deps.clock() - state["started"] < TIME_BUDGET_S):
            pending = [followup]
        record = {"sufficient": sufficient, "followup_query": followup or None, "will_retrieve": bool(pending),
                  "parse_error": info.get("parse_error", False)}
        return {"pending": pending, "queries": state["queries"] + pending,
                "assessments": state["assessments"] + [record],
                "llm_calls": _bump(state["llm_calls"], "assess"),
                "parse_errors": _bump(state["parse_errors"], "assess") if info.get("parse_error")
                else state["parse_errors"],
                "events": state["events"] + [{"node": "assess", **record}]}

    def route(state: GraphState) -> str:
        return "retrieve" if state["pending"] else "synthesize"

    async def synthesize(state: GraphState) -> dict:
        context = fused(state, CONTEXT_BUDGET)
        sources = "\n\n".join(f"[S{i}] ({deps.source_label(c)})\n{c['text']}" for i, c in enumerate(context, start=1))
        t0 = deps.clock()
        gen = await deps.generate(SYNTH_SYSTEM.format(sources=sources), state["question"])
        gen = {**gen, "started_after_s": t0 - state["started"]}
        return {"context": [{**c, "rank": i, "source_label": deps.source_label(c)} for i, c in enumerate(context, 1)],
                "generation": gen, "llm_calls": _bump(state["llm_calls"], "synthesize"),
                "events": state["events"] + [{"node": "synthesize", "sources": len(context)}]}

    def verify(state: GraphState) -> dict:
        context = state["context"]
        cites = parse_citations(state["generation"].get("text", ""), len(context))
        cites["cited_chunk_ids"] = [context[i - 1]["chunk_id"] for i in cites["valid_sources"]]
        cites["uncited_answer"] = cites["markers"] == 0 and not cites["abstained"]
        return {"citations": cites, "events": state["events"] + [{"node": "verify", **{
            k: cites[k] for k in ("markers", "invalid_sources", "uncited_answer")}}]}

    graph = StateGraph(GraphState)
    for name, fn in (("plan", plan), ("retrieve", retrieve), ("assess", assess), ("synthesize", synthesize),
                     ("verify", verify)):
        graph.add_node(name, fn)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "assess")
    graph.add_conditional_edges("assess", route, {"retrieve": "retrieve", "synthesize": "synthesize"})
    graph.add_edge("synthesize", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


async def run_graph(deps: GraphDeps, question: str) -> dict:
    """Run once; returns the final state plus summary fields used by the harness trace."""
    app = build_graph(deps)
    started = deps.clock()
    # plan, (retrieve, assess) x MAX_ROUNDS, synthesize, verify  → well under this limit
    state = await app.ainvoke({"question": question, "started": started, "events": [], "llm_calls": {},
                               "parse_errors": {}}, config={"recursion_limit": 4 + 2 * MAX_ROUNDS + 2})
    state["elapsed_s"] = deps.clock() - started
    return state
