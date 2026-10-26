import math

import pytest

import metrics as M

GOLD = [
    {"paper": "octo", "page": 3, "quote": "trained on 800k robot trajectories", "facet": "dataset"},
    {"paper": "octo", "page": 5, "quote": "uses a diffusion action head", "facet": "method"},
]


def chunk(paper, pages, text):
    return {"paper_id": paper, "pages": pages, "text": text}


# --- normalization / matching ---------------------------------------------------
def test_normalize_handles_ligatures_hyphen_breaks_case_and_whitespace():
    assert M.normalize("Eﬃcient  Pre-\ntrained\tModel") == "efficient pretrained model"


def test_normalize_treats_line_break_hyphens_like_real_hyphens():
    # PDF line breaks split words; the paper's own spelling must still match.
    assert M.normalize("an open- source 3 billion") == M.normalize("an open-source 3 billion")
    assert M.normalize("dif- ferent robot") == M.normalize("different robot")
    assert M.normalize("RT-1-X and 80-90%") == "rt1x and 8090%"
    assert M.normalize("a - b") == "a - b"  # a spaced dash is punctuation, not a word hyphen


def test_evidence_hit_substring_and_coverage_threshold():
    text = "Octo is trained on 800k robot trajectories from Open X."
    assert M.evidence_hit("trained on 800k robot trajectories", text)
    # Quote cut by a chunk boundary: 4 of 5 tokens present = 0.8 coverage -> hit at default threshold
    assert M.evidence_hit("trained on 800k robot trajectories", "is trained on 800k robot")
    assert not M.evidence_hit("trained on 800k robot trajectories", "trained on robot")  # 3/5
    assert M.token_coverage("", "anything") == 0.0


def test_evidence_requires_matching_paper():
    c = chunk("openvla", [3], "trained on 800k robot trajectories")
    assert not M.chunk_matches_evidence(c, GOLD[0])


# --- retrieval metrics ------------------------------------------------------------
RETRIEVED = [
    chunk("rt1", [2], "unrelated"),                                   # rank 1 — irrelevant
    chunk("octo", [3], "Octo is trained on 800k robot trajectories"),  # rank 2 — gold 0
    chunk("octo", [5, 6], "figure caption without the quote"),       # rank 3 — page match only
    chunk("octo", [9], "more text"),                                  # rank 4
]


def test_evidence_recall_and_page_recall():
    assert M.evidence_recall_at_k(GOLD, RETRIEVED, 5) == 0.5
    assert M.evidence_recall_at_k(GOLD, RETRIEVED, 1) == 0.0
    assert M.page_recall_at_k(GOLD, RETRIEVED, 5) == 1.0
    assert M.page_recall_at_k(GOLD, RETRIEVED, 2) == 0.5


def test_precision_uses_number_actually_returned():
    assert M.precision_at_k(GOLD, RETRIEVED, 5, "evidence") == pytest.approx(1 / 4)
    assert M.precision_at_k(GOLD, RETRIEVED, 5, "page") == pytest.approx(2 / 4)
    assert M.precision_at_k(GOLD, [], 5) == 0.0


def test_reciprocal_rank():
    assert M.reciprocal_rank(GOLD, RETRIEVED, "evidence") == 0.5
    assert M.reciprocal_rank(GOLD, RETRIEVED[:1], "evidence") == 0.0


def test_paper_recall_and_facet_coverage():
    assert M.paper_recall_at_k(["octo", "openvla"], RETRIEVED, 5) == 0.5
    assert M.facet_coverage_at_k(GOLD, RETRIEVED, 5) == 0.5  # dataset covered, method not


def test_metrics_are_undefined_for_negative_questions():
    for fn in (M.evidence_recall_at_k, M.page_recall_at_k, M.facet_coverage_at_k):
        assert fn([], RETRIEVED, 5) is None
    assert M.precision_at_k([], RETRIEVED, 5) is None
    assert M.reciprocal_rank([], RETRIEVED) is None
    assert M.paper_recall_at_k([], RETRIEVED, 5) is None


def test_alternative_locations_count_as_coverage():
    ev = {"paper": "pi0", "page": 11, "quote": "10,000 hours of dexterous manipulation data", "facet": "dataset",
          "alternatives": [{"page": 2, "quote": "pre-trained on diverse data from 7 distinct robot configurations"}]}
    alt_hit = [chunk("pi0", [2], "Our model is pre-trained on diverse data from 7 distinct robot configurations and 68 tasks")]
    assert M.evidence_recall_at_k([ev], alt_hit, 5) == 1.0
    assert M.page_recall_at_k([ev], alt_hit, 5) == 1.0
    miss = [chunk("pi0", [5], "unrelated architecture details")]
    assert M.evidence_recall_at_k([ev], miss, 5) == 0.0 and M.page_recall_at_k([ev], miss, 5) == 0.0
    assert [v["page"] for v in M.variants(ev)] == [11, 2]


def test_score_retrieval_keys():
    out = M.score_retrieval({"evidence": GOLD, "papers": ["octo"]}, RETRIEVED, ks=(5,))
    assert out["evidence_recall@5"] == 0.5 and out["paper_recall@5"] == 1.0


# --- answer heuristics -------------------------------------------------------------
@pytest.mark.parametrize("answer, expected", [
    ("The provided documents do not contain information about this.", True),
    ("I cannot find that in the context.", True),
    ("Octo was trained on 800k trajectories.", False),
])
def test_abstention_heuristic(answer, expected):
    assert M.abstention_detected(answer) is expected


def test_keyword_recall_and_citation_markers():
    assert M.keyword_recall("Trained on 800K trajectories with 93M params", ["800k", "93M", "Octo"]) == pytest.approx(2 / 3)
    assert M.keyword_recall("anything", []) is None
    assert M.citation_markers("See [S1] and (p. 4); also page 7.") == 3
    assert M.citation_markers("No citations here.") == 0


def test_score_answer_negative_vs_positive():
    neg = M.score_answer({"type": "negative"}, "The documents do not mention this.")
    assert neg["abstention_correct"] is True and neg["keyword_recall"] is None and neg["false_abstention"] is None
    pos = M.score_answer({"type": "lookup", "answer_keywords": ["800k"]}, "800k trajectories")
    assert pos["false_abstention"] is False and pos["abstention_correct"] is None and pos["keyword_recall"] == 1.0


# --- aggregation -------------------------------------------------------------------------
def test_percentile_linear_interpolation():
    values = [1, 2, 3, 4]
    assert M.percentile(values, 50) == pytest.approx(2.5)
    assert M.percentile(values, 95) == pytest.approx(3.85)
    assert M.percentile([7.0], 95) == 7.0
    assert M.percentile([], 50) is None
    assert M.percentile([None, float("nan"), 2.0], 50) == 2.0


def test_mean_defined_ignores_none_and_reports_n():
    assert M.mean_defined([1.0, None, 0.0]) == (0.5, 2)
    value, n = M.mean_defined([None, None])
    assert value is None and n == 0
    assert not math.isnan(M.mean_defined([0.0])[0])
