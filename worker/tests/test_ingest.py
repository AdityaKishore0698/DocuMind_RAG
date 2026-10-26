"""Page/section-aware ingestion: pure functions, and `ingest_bytes` against a throwaway Postgres
(WORKER_TEST_DATABASE_URL, loopback only; skipped when unset)."""

import importlib.util
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

WORKER = Path(__file__).resolve().parents[1]
REPO = WORKER.parent
sys.path.insert(0, str(WORKER))

TEST_DB = os.getenv("WORKER_TEST_DATABASE_URL")
if TEST_DB:
    assert urlparse(TEST_DB).hostname in {"127.0.0.1", "localhost"}, "tests only run against a local database"
    os.environ["DATABASE_URL"] = TEST_DB

from ingest import (Extracted, Page, canonical_heading, chunk_pages, extract_pages, find_headings,  # noqa: E402
                    first_page_title, paper_metadata)

SAMPLE_PDF = REPO / "evaluation" / "data" / "pdfs" / "diffusion_policy.pdf"


@pytest.mark.parametrize("line,expected", [
    ("1 Introduction", "Introduction"), ("I. I NTRODUCTION", "Introduction"), ("2.1 Related Work", "Related Work"),
    ("RELATEDWORK", "Related Work"), ("Limitations and Future Work", "Limitations"), ("5. Conclusions", "Conclusion"),
    ("the introduction of a new method", None), ("A Model", None), ("References", "References")])
def test_canonical_headings(line, expected):
    assert canonical_heading(line) == expected


def test_custom_numbered_headings_and_run_in():
    names = [n for _, n in find_headings("text\nIII. THE OCTO MODEL\nmore\n4 Octo Model Design\nend.II. R ELATED WORK\n")]
    assert names == ["The Octo Model", "Octo Model Design", "Related Work"]


def test_chunks_carry_page_and_section_across_pages():
    pages = [Page(1, "Title\nAbstract\nWe study X.\n1 Introduction\n" + "intro words " * 120),
             Page(2, "continued intro " * 10 + "\n2 Method\nOur method is Y."),
             Page(3, "")]
    chunks = chunk_pages(pages)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].section_heading is None and chunks[0].page_number == 1
    page2 = [c for c in chunks if c.page_number == 2]
    assert page2[0].section_heading == "Introduction" and page2[-1].section_heading == "Method"
    assert all(len(c.content) <= 1000 for c in chunks)


def test_plain_text_has_no_pages_and_nul_is_stripped():
    extracted = extract_pages(b"hello\x00 world", "notes.txt")
    assert extracted.pages[0].page_number is None and extracted.pages[0].text == "hello world"


def test_metadata_sources():
    ex = Extracted(pages=[Page(1, "A Great Paper Title\nAuthors\nAbstract\narXiv:2303.04137v5")], pdf_title="paper.pdf")
    meta = paper_metadata(ex, chunk_pages(ex.pages))
    assert meta["title"] == "A Great Paper Title" and meta["title_source"] == "first_page_heuristic"
    assert meta["year"] == 2023 and meta["year_source"] == "arxiv_identifier"


@pytest.mark.parametrize("text,title", [
    ("Diffusion Policy: Visuomotor Policy\nLearning via Action Diffusion.\nCheng Chi1, Zhenjia Xu1\nAbstract",
     "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"),
    ("Scaling Cross-Embodied Learning: One Policy for\nManipulation, Navigation and Aviation\nRia Doshi1Homer Walke1",
     "Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation and Aviation"),
    ("Octo: An Open-Source Generalist Robot Policy\nOcto Model Team\nDibya Ghosh1", "Octo: An Open-Source Generalist Robot Policy"),
    ("https://robotics-transformer2.github.io\n2023-8-1\nRT-2: Models Transfer\nWeb Knowledge to Robotic Control\nA. Brohan, N. Brown",
     "RT-2: Models Transfer Web Knowledge to Robotic Control"),
    ("Preprint\nRT-1: R OBOTICS TRANSFORMER\nFOR REAL-WORLD CONTROL\n1Anthony Brohan",
     "RT-1: ROBOTICS TRANSFORMER FOR REAL-WORLD CONTROL"),
])
def test_first_page_title(text, title):
    assert first_page_title(text) == title


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="evaluation corpus not present")
def test_real_paper_extraction():
    ex = extract_pages(SAMPLE_PDF.read_bytes(), SAMPLE_PDF.name)
    chunks = chunk_pages(ex.pages)
    meta = paper_metadata(ex, chunks)
    assert len(ex.pages) > 5 and len(chunks) > 50
    assert {"Introduction", "Related Work", "Conclusion"} <= set(meta["sections"])
    assert all(c.page_number for c in chunks)


# --- database integration ---------------------------------------------------------------------------------
needs_db = pytest.mark.skipif(not TEST_DB, reason="WORKER_TEST_DATABASE_URL not set")


@pytest.fixture(scope="module")
def db():
    from sqlalchemy import text

    from database import Base, SessionLocal, engine
    import models  # noqa: F401

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    s = SessionLocal()
    yield s
    s.close()
    Base.metadata.drop_all(engine)


def fake_embed(texts):
    return [[0.1] * 768 for _ in texts]


@needs_db
def test_ingest_is_idempotent_and_stores_provenance(db):
    import tasks
    from models import Document, DocumentChunk

    doc = Document(user_id=1, filename="paper.txt", status="processing")
    db.add(doc)
    db.commit()
    body = ("A Paper Title Here\nAbstract\nSummary.\n1 Introduction\n" + "words " * 400 + "\n2 Method\nDetails.").encode()
    n1 = tasks.ingest_bytes(db, doc, body, embed=fake_embed)
    n2 = tasks.ingest_bytes(db, doc, body, embed=fake_embed)
    rows = db.query(DocumentChunk).filter_by(document_id=doc.id).order_by(DocumentChunk.chunk_index).all()
    assert n1 == n2 == len(rows)
    assert [r.chunk_index for r in rows] == list(range(len(rows)))
    assert rows[-1].section_heading == "Method" and rows[0].page_number is None
    db.refresh(doc)
    assert doc.status == "completed" and doc.title == "A Paper Title Here" and doc.error is None
    from inference import embedding_space
    assert doc.embedding_model == embedding_space()   # the space is recorded with the vectors


@needs_db
def test_failed_embedding_leaves_previous_chunks_untouched(db):
    import tasks
    from models import Document, DocumentChunk

    doc = Document(user_id=1, filename="p.txt", status="processing")
    db.add(doc)
    db.commit()
    tasks.ingest_bytes(db, doc, b"Some text worth keeping.", embed=fake_embed)

    def broken(texts):
        raise RuntimeError("provider down")

    with pytest.raises(RuntimeError):
        tasks.ingest_bytes(db, doc, b"New text.", embed=broken)
    db.rollback()
    assert [r.content for r in db.query(DocumentChunk).filter_by(document_id=doc.id)] == ["Some text worth keeping."]


@needs_db
def test_empty_document_fails_with_a_reason(db, monkeypatch):
    import tasks
    from models import Document

    doc = Document(user_id=1, filename="blank.txt", status="processing")
    db.add(doc)
    db.commit()
    monkeypatch.setattr(tasks, "get_supabase", lambda: type("S", (), {"storage": type("St", (), {
        "from_": staticmethod(lambda b: type("B", (), {"download": staticmethod(lambda p: b"   ")})())})()})())
    assert "failed" in tasks.process_document(doc.id, "1/x/blank.txt")
    db.expire_all()
    doc = db.get(Document, doc.id)
    assert doc.status == "failed" and "no extractable text" in doc.error


def _storage_returning(data: bytes):
    return lambda: type("S", (), {"storage": type("St", (), {
        "from_": staticmethod(lambda b: type("B", (), {"download": staticmethod(lambda p: data)})())})()})()


@needs_db
def test_processing_keeps_the_chat_and_reprocessing_replaces_chunks(db, monkeypatch):
    import tasks
    from models import Document, DocumentChunk

    doc = Document(user_id=1, session_id=42, filename="chat.txt", status="uploaded")
    db.add(doc)
    db.commit()
    monkeypatch.setattr(tasks, "get_supabase", _storage_returning(b"Some chat-scoped text. " * 80))
    monkeypatch.setattr(tasks, "get_embeddings", fake_embed)
    monkeypatch.setattr(tasks.ingest_bytes, "__defaults__", (fake_embed,))
    assert "processed" in tasks.process_document(doc.id, "1/x/chat.txt", 42)
    first = db.query(DocumentChunk).filter_by(document_id=doc.id).count()
    assert "processed" in tasks.process_document(doc.id, "1/x/chat.txt", 42)      # re-process
    db.expire_all()
    assert db.query(DocumentChunk).filter_by(document_id=doc.id).count() == first > 0   # replaced, not duplicated
    assert db.get(Document, doc.id).session_id == 42 and db.get(Document, doc.id).status == "completed"


@needs_db
def test_a_task_for_another_chat_is_not_ingested(db, monkeypatch):
    import tasks
    from models import Document, DocumentChunk

    doc = Document(user_id=1, session_id=7, filename="other.txt", status="uploaded")
    db.add(doc)
    db.commit()
    monkeypatch.setattr(tasks, "get_supabase", lambda: pytest.fail("must not download"))
    assert "skipped" in tasks.process_document(doc.id, "1/x/other.txt", 8)
    db.expire_all()
    assert db.query(DocumentChunk).filter_by(document_id=doc.id).count() == 0
    assert db.get(Document, doc.id).session_id == 7


def test_worker_and_api_models_map_the_same_columns(monkeypatch):
    """worker/models.py must stay in sync with api/models (shared tables)."""
    monkeypatch.syspath_prepend(str(REPO / "api"))
    spec = importlib.util.spec_from_file_location("api_document_models", REPO / "api" / "models" / "document.py")
    api_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api_models)
    import models as worker_models

    def cols(model):  # unbounded String and Text are the same Postgres type
        return {c.name: type(c.type).__name__.replace("Text", "String") for c in model.__table__.columns}

    for name in ("Document", "DocumentChunk"):
        api_cols, worker_cols = cols(getattr(api_models, name)), cols(getattr(worker_models, name))
        assert worker_cols == api_cols, name
