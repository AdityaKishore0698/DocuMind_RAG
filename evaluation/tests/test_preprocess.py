"""Regression tests for the evaluation-only NUL stripping (preprocess.py)."""

import os

import psycopg2
import pytest
from psycopg2.extensions import QuotedString

import baseline_a
import db
import preprocess
from common import load_corpus, load_pages
from provenance import concat_with_spans, map_chunks

RAW_PAGES = [
    "Methods. We sample noise x ~ N\x00 0, I and denoise. " * 25,   # NUL inside maths
    "\x00\x00",                                                       # page that is ONLY NULs
    "",                                                               # truly empty page (skipped)
    "Results. weights w = exp(\x00m i) decay over time. " * 25,
]


def _adapt(text: str) -> bytes:
    """psycopg2 literal adaptation as used by a UTF-8 connection (the NUL check is
    encoding-independent; without a connection psycopg2 would default to latin-1)."""
    quoted = QuotedString(text)
    quoted.encoding = "utf-8"
    return quoted.getquoted()


def _production_concat(pages):
    """worker/tasks.py extract_text's concatenation rule, on raw page texts."""
    text = ""
    for extracted in pages:
        if extracted:
            text += extracted + "\n"
    return text


def test_raw_nul_text_is_rejected_by_the_postgres_driver():
    with pytest.raises(ValueError, match="NUL"):
        _adapt(RAW_PAGES[0])


def test_stripped_text_is_accepted_by_the_postgres_driver():
    _adapt(preprocess.strip_nul(RAW_PAGES[0]))


def test_strip_nul_removes_only_nul():
    assert preprocess.strip_nul("a\x00b\x00") == "ab"
    assert preprocess.strip_nul("no nul here ∼ ﬁ") == "no nul here ∼ ﬁ"
    assert preprocess.nul_count("x\x00y\x00\x00") == 3


def test_page_spans_and_chunks_stay_consistent_after_stripping():
    text, spans = concat_with_spans(RAW_PAGES, preprocess.strip_nul)
    # identical to stripping production's concatenated text
    assert text == preprocess.strip_nul(_production_concat(RAW_PAGES))
    assert "\x00" not in text
    # inclusion follows the RAW text: the all-NUL page 2 is kept (as empty), page 3 is skipped
    assert [s.page_number for s in spans] == [1, 2, 4]
    page2 = spans[1]
    assert page2.start == page2.end
    for span, raw in zip(spans, [RAW_PAGES[0], RAW_PAGES[1], RAW_PAGES[3]]):
        assert text[span.start:span.end] == preprocess.strip_nul(raw)
    chunks = baseline_a.split_text(text)
    prov = map_chunks(text, spans, chunks)
    assert all(p.mapped and p.pages for p in prov)
    assert all(text[p.char_start:p.char_end] == p.text for p in prov)
    assert any(4 in p.pages and "exp(m i)" in p.text for p in prov)
    assert not any(2 in p.pages for p in prov)  # an empty page can never be cited


@pytest.mark.parametrize("paper_id", ["octo", "diffusion_policy", "act_aloha"])
def test_real_affected_papers_map_cleanly(paper_id):
    papers = {p.id: p for p in load_corpus()}
    paper = papers[paper_id]
    if not paper.pages_path.exists():
        pytest.skip("extracted pages not present (run download_corpus.py + run_eval.py extract)")
    pages = load_pages(paper)
    assert any("\x00" in p for p in pages), "fixture expectation: this paper contains NUL characters"
    text, spans = concat_with_spans(pages, preprocess.strip_nul)
    assert text == preprocess.strip_nul(_production_concat(pages))
    prov = map_chunks(text, spans, baseline_a.split_text(text))
    assert all(p.mapped for p in prov) and not any("\x00" in p.text for p in prov)
    for p in prov:
        _adapt(p.text)  # every chunk is storable


def test_real_database_round_trip_when_available():
    url = os.environ.get("EVAL_DATABASE_URL", db.DEFAULT_EVAL_DATABASE_URL)
    try:
        db.assert_local_eval_database(url)
        conn = psycopg2.connect(url, connect_timeout=2)
    except Exception:  # noqa: BLE001 — optional integration check
        pytest.skip("local evaluation database not reachable")
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE nul_check (content TEXT)")
            with pytest.raises(ValueError, match="NUL"):
                cur.execute("INSERT INTO nul_check VALUES (%s)", (RAW_PAGES[0],))
            cur.execute("INSERT INTO nul_check VALUES (%s)", (preprocess.strip_nul(RAW_PAGES[0]),))
            cur.execute("SELECT count(*) FROM nul_check")
            assert cur.fetchone()[0] == 1
    finally:
        conn.rollback()
        conn.close()
