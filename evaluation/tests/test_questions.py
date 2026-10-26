"""Structural checks on the committed corpus/questions (no PDFs or network needed).
Quote-on-page checks need the PDFs: `run_eval.py validate-annotations`."""

from common import load_corpus, load_questions, validate_questions


def test_corpus_loads_with_unique_ids():
    papers = load_corpus(enabled_only=False)
    assert len({p.id for p in papers}) == len(papers)
    assert all(p.license.startswith("http") for p in papers)
    assert 8 <= len(load_corpus()) <= 15


def test_questions_file_is_schema_valid():
    corpus_ids = {p.id for p in load_corpus(enabled_only=False)}
    questions = load_questions()
    assert questions, "questions.jsonl must not be empty"
    assert validate_questions(questions, corpus_ids) == []


def test_question_mix_covers_required_types():
    types = {q["type"] for q in load_questions()}
    assert {"lookup", "cross_paper", "negative"} <= types
    assert types & {"method", "dataset", "result", "limitation"}


def test_answer_keywords_appear_in_the_gold_answer():
    import metrics as M

    for q in load_questions():
        if q.get("answer_keywords"):
            assert M.keyword_recall(q["gold_answer"], q["answer_keywords"]) == 1.0, q["id"]


def test_validator_catches_common_mistakes():
    corpus = {"octo", "rt1"}
    bad = [
        {"id": "x1", "status": "draft", "type": "negative", "papers": ["octo"], "question": "q?",
         "gold_answer": "", "evidence": []},
        {"id": "x2", "status": "draft", "type": "cross_paper", "papers": ["octo"], "question": "q?",
         "gold_answer": "a", "evidence": [{"paper": "octo", "page": 1, "quote": "four words in quote", "facet": "method"}]},
        {"id": "x3", "status": "weird", "type": "lookup", "papers": ["nope"], "question": "q?",
         "gold_answer": "a", "evidence": [{"paper": "nope", "page": 0, "quote": "too short", "facet": "vibes"}]},
        {"id": "x3", "status": "draft"},
    ]
    errors = "\n".join(validate_questions(bad, corpus))
    for expected in ("negative questions must have no papers", "needs >= 2 papers", "unknown status",
                     "not in corpus", "page must be a positive int", "unknown facet", "quote too short",
                     "duplicate id", "missing field"):
        assert expected in errors, expected
