"""Deterministic evaluation metrics (pure functions — no I/O, no API calls).

Conventions
-----------
* A *gold evidence item* is ``{"paper", "page", "quote", "facet"}`` from questions.jsonl.
* A *retrieved chunk* is ``{"paper_id", "pages", "text", ...}`` in rank order.
* Evidence-level relevance: the chunk comes from the gold paper and contains
  the gold quote — as a normalized substring, or with token coverage >= the
  threshold (quotes can straddle a chunk boundary).
* Page-level relevance (lenient): the chunk comes from the gold paper and
  overlaps the gold page.
* Metrics return ``None`` when undefined for a question (e.g. recall for a
  negative question). Aggregates ignore ``None`` and report how many
  questions contributed, so "not measured" is never reported as 0.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

COVERAGE_THRESHOLD = 0.8

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*%?")
_WORD_HYPHEN = re.compile(r"(?<=\w)-\s*(?=\w)")


def normalize(text: str) -> str:
    """NFKC (fixes ligatures like 'ﬁ'), drop word-internal hyphens, lowercase,
    collapse whitespace.

    PDF extraction splits words at line breaks ('open- source', 'dif- ferent'),
    so a hyphen is removed whether or not whitespace follows it. Applied to
    both sides of every comparison, 'open-source', 'open- source' and
    'opensource' all compare equal."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _WORD_HYPHEN.sub("", text)
    return _WS.sub(" ", text).strip().lower()


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


def token_coverage(quote: str, text: str) -> float:
    """Fraction of the quote's tokens (with multiplicity) present in ``text``."""
    q = Counter(tokens(quote))
    if not q:
        return 0.0
    t = Counter(tokens(text))
    hit = sum(min(n, t[tok]) for tok, n in q.items())
    return hit / sum(q.values())


def evidence_hit(quote: str, chunk_text: str, threshold: float = COVERAGE_THRESHOLD) -> bool:
    nq, nc = normalize(quote), normalize(chunk_text)
    if nq and nq in nc:
        return True
    return token_coverage(quote, chunk_text) >= threshold


def variants(ev: dict) -> list[dict]:
    """The main location plus any annotated alternative locations of the same fact
    (``alternatives: [{"page", "quote"}]``, same paper). Covering any one counts."""
    return [{"page": ev["page"], "quote": ev["quote"]}, *ev.get("alternatives", [])]


def chunk_matches_evidence(chunk: dict, ev: dict, threshold: float = COVERAGE_THRESHOLD) -> bool:
    if chunk.get("paper_id") != ev["paper"]:
        return False
    return any(evidence_hit(v["quote"], chunk.get("text", ""), threshold) for v in variants(ev))


def chunk_matches_page(chunk: dict, ev: dict) -> bool:
    if chunk.get("paper_id") != ev["paper"]:
        return False
    pages = chunk.get("pages") or []
    return any(v["page"] in pages for v in variants(ev))


def _top(retrieved: list[dict], k: int) -> list[dict]:
    return retrieved[: max(0, k)]


def evidence_recall_at_k(gold: list[dict], retrieved: list[dict], k: int) -> float | None:
    if not gold:
        return None
    top = _top(retrieved, k)
    covered = sum(any(chunk_matches_evidence(c, ev) for c in top) for ev in gold)
    return covered / len(gold)


def page_recall_at_k(gold: list[dict], retrieved: list[dict], k: int) -> float | None:
    """Share of gold evidence items whose page (or an alternative page) is
    overlapped by a top-k chunk from the right paper."""
    if not gold:
        return None
    top = _top(retrieved, k)
    covered = sum(any(chunk_matches_page(c, ev) for c in top) for ev in gold)
    return covered / len(gold)


def precision_at_k(gold: list[dict], retrieved: list[dict], k: int, mode: str = "evidence") -> float | None:
    """Relevant chunks among the top-k actually returned (denominator = len(top-k))."""
    if not gold:
        return None
    top = _top(retrieved, k)
    if not top:
        return 0.0
    match = chunk_matches_evidence if mode == "evidence" else chunk_matches_page
    relevant = sum(any(match(c, ev) for ev in gold) for c in top)
    return relevant / len(top)


def reciprocal_rank(gold: list[dict], retrieved: list[dict], mode: str = "evidence") -> float | None:
    if not gold:
        return None
    match = chunk_matches_evidence if mode == "evidence" else chunk_matches_page
    for rank, chunk in enumerate(retrieved, start=1):
        if any(match(chunk, ev) for ev in gold):
            return 1.0 / rank
    return 0.0


def paper_recall_at_k(gold_papers: list[str], retrieved: list[dict], k: int) -> float | None:
    """Fraction of the question's gold papers that appear at all in the top-k."""
    papers = set(gold_papers)
    if not papers:
        return None
    present = {c.get("paper_id") for c in _top(retrieved, k)}
    return len(papers & present) / len(papers)


def facet_coverage_at_k(gold: list[dict], retrieved: list[dict], k: int) -> float | None:
    """Evidence coverage by facet: a facet counts as covered when at least one of
    its gold evidence items is matched in the top-k."""
    if not gold:
        return None
    top = _top(retrieved, k)
    facets: dict[str, bool] = {}
    for ev in gold:
        hit = any(chunk_matches_evidence(c, ev) for c in top)
        facets[ev["facet"]] = facets.get(ev["facet"], False) or hit
    return sum(facets.values()) / len(facets)


# --- answer-level heuristics (clearly labelled as heuristics in reports) ------
_ABSTAIN = re.compile(
    r"(does not|doesn't|do not|don't|cannot|can't|unable to|no information|not (?:mentioned|provided|"
    r"specified|stated|discussed|covered|available|contain|include)|isn't (?:mentioned|provided)|"
    r"not enough (?:information|context)|context (?:does not|doesn't|provided does not))",
    re.IGNORECASE,
)


def abstention_detected(answer: str) -> bool:
    """Heuristic: did the answer say the documents do not contain the answer?"""
    return bool(_ABSTAIN.search(answer or ""))


def keyword_recall(answer: str, keywords: list[str]) -> float | None:
    """Heuristic: fraction of expected key facts (normalized substrings) present."""
    if not keywords:
        return None
    na = normalize(answer)
    return sum(normalize(k) in na for k in keywords) / len(keywords)


_CITATION = re.compile(r"\[(?:S?\d+|source[^\]]*)\]|\(p(?:age|\.)?\s*\d+\)|\bpage\s+\d+\b|\bp\.\s*\d+\b",
                       re.IGNORECASE)


def citation_markers(answer: str) -> int:
    """Count citation-like markers. System A is not prompted to cite, so this
    is reported for information only — citation *support* needs System B/C."""
    return len(_CITATION.findall(answer or ""))


# --- aggregation -------------------------------------------------------------------
def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile (numpy's default 'linear'); None if empty."""
    data = sorted(v for v in values if v is not None and not math.isnan(v))
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * (p / 100.0)
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def mean_defined(values: list[float | None]) -> tuple[float | None, int]:
    """Mean over defined values and the number of contributing items."""
    defined = [v for v in values if v is not None]
    return (sum(defined) / len(defined) if defined else None), len(defined)


def score_retrieval(question: dict, retrieved: list[dict], ks: tuple[int, ...] = (5,)) -> dict:
    """All retrieval metrics for one question."""
    gold = question.get("evidence", [])
    out: dict = {}
    for k in ks:
        out[f"evidence_recall@{k}"] = evidence_recall_at_k(gold, retrieved, k)
        out[f"page_recall@{k}"] = page_recall_at_k(gold, retrieved, k)
        out[f"evidence_precision@{k}"] = precision_at_k(gold, retrieved, k, "evidence")
        out[f"page_precision@{k}"] = precision_at_k(gold, retrieved, k, "page")
        out[f"paper_recall@{k}"] = paper_recall_at_k(question.get("papers", []), retrieved, k)
        out[f"facet_coverage@{k}"] = facet_coverage_at_k(gold, retrieved, k)
    out["evidence_mrr"] = reciprocal_rank(gold, retrieved, "evidence")
    out["page_mrr"] = reciprocal_rank(gold, retrieved, "page")
    return out


def score_answer(question: dict, answer: str) -> dict:
    is_negative = question.get("type") == "negative"
    abstained = abstention_detected(answer)
    return {
        "keyword_recall": None if is_negative else keyword_recall(answer, question.get("answer_keywords", [])),
        "abstained": abstained,
        "abstention_correct": abstained if is_negative else None,
        "false_abstention": (abstained if not is_negative else None),
        "citation_markers": citation_markers(answer),
    }


def score_citations(question: dict, citations: dict, context: list[dict]) -> dict:
    """Citation metrics for one graph answer.

    * citation_validity          — share of distinct cited source numbers that exist in the context (None if none).
    * citation_support_precision — share of validly cited chunks containing a gold evidence quote of this
                                   question (None for questions without gold evidence or without citations).
    * answer_has_citations       — 1 if any marker, else 0 (None for negative questions, where abstaining
                                   without citations is correct).
    """
    valid, invalid = citations.get("valid_sources", []), citations.get("invalid_sources", [])
    total = len(valid) + len(invalid)
    gold = question.get("evidence", [])
    cited = [context[i - 1] for i in valid if 1 <= i <= len(context)]
    support = None
    if gold and cited:
        support = sum(any(chunk_matches_evidence(c, ev) for ev in gold) for c in cited) / len(cited)
    return {"citation_validity": (len(valid) / total) if total else None,
            "citation_support_precision": support,
            "answer_has_citations": (1.0 if citations.get("markers") else 0.0) if gold else None}
