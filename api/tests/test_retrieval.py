"""Feature-flagged retrieval (core/retrieval.py) + migration 001_bm25_hybrid, against the test Postgres.

Covers: vector mode == legacy query; BM25 scores == an independent Python implementation; per-user isolation of
candidates AND statistics; document filter; empty/tiny libraries; no-lexeme queries; timeout and error fallback with
savepoint recovery and content-free logs; trigger lifecycle (insert, update, delete, failed ingestion); migration
up/down/up, idempotency, invalid-index recovery; startup validation; /chat end to end in hybrid mode.
"""

import importlib.util
import json
import logging
import math
import os
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

import core.retrieval as R
from models.document import Document, DocumentChunk
from models.user import User

from conftest import TEST_DATABASE_URL, TestingSessionLocal, engine

_spec = importlib.util.spec_from_file_location(
    "bm25_migration", Path(__file__).resolve().parents[2] / "scripts" / "migrations" / "bm25.py")
migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migration)

DIM = 768


def vec(*hot: float) -> list[float]:
    v = [0.0] * DIM
    for i, x in enumerate(hot):
        v[i] = x
    return v


@pytest.fixture(scope="module", autouse=True)
def migrated():
    conn = migration.connect(TEST_DATABASE_URL)
    migration.up(conn, log=lambda *_: None)
    conn.close()
    yield


@pytest.fixture
def db():
    s = TestingSessionLocal()
    yield s
    s.rollback()
    s.close()


@pytest.fixture
def make_library(db):
    """Create a user with documents {name: [(content, embedding), ...]}; cleans up afterwards."""
    created = []

    def _make(docs: dict) -> tuple[int, dict]:
        user = User(supabase_uid=str(uuid.uuid4()), email="r@example.com")
        db.add(user)
        db.commit()
        ids = {}
        for name, chunks in docs.items():
            d = Document(user_id=user.id, filename=name, status="completed")
            db.add(d)
            db.flush()
            for content, emb in chunks:
                db.add(DocumentChunk(document_id=d.id, content=content, embedding=emb))
            ids[name] = d.id
        db.commit()
        created.append(user.id)
        return user.id, ids

    yield _make
    db.rollback()
    for uid in created:
        db.execute(text("DELETE FROM document_chunks WHERE document_id IN (SELECT id FROM documents WHERE user_id = :u)"),
                   {"u": uid})
        db.execute(text("DELETE FROM documents WHERE user_id = :u"), {"u": uid})
        db.execute(text("DELETE FROM users WHERE id = :u"), {"u": uid})
    db.commit()


@pytest.fixture
def hybrid(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid_bm25")
    R.FALLBACKS.clear()


LIB = {
    "octo.pdf": [("Octo is trained on 800k robot trajectories from Open X-Embodiment.", vec(0.2, 1.0)),
                 ("The robot policy uses a robot arm and a robot gripper for robot tasks.", vec(1.0, 0.1)),
                 ("Diffusion action heads model multimodal action distributions.", vec(0.9, 0.3))],
    "aloha.pdf": [("ALOHA is a low-cost bimanual teleoperation kit costing under 20k USD.", vec(0.1, 0.1, 1.0)),
                  ("ACT predicts action chunks with a CVAE.", vec(0.5, 0.5, 0.5))],
}


# --- configuration --------------------------------------------------------------------------------------
def test_mode_defaults_to_vector_and_rejects_unknown(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
    assert R.retrieval_mode() == "vector"
    assert R.validate_configuration(engine) == "vector"
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25-only")
    with pytest.raises(R.RetrievalConfigError):
        R.validate_configuration(engine)


def test_vector_mode_is_exactly_the_legacy_query(db, make_library, monkeypatch):
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
    uid, _ = make_library(LIB)
    q = vec(1.0, 0.2)
    legacy = db.query(DocumentChunk).join(Document).filter(Document.user_id == uid).order_by(
        DocumentChunk.embedding.cosine_distance(q)).limit(5).all()
    got = R.retrieve_chunks(db, user_id=uid, query_text="robot policy", query_embedding=q)
    assert [c.id for c in got] == [c.id for c in legacy] and len(got) == 5


# --- scoring correctness ------------------------------------------------------------------------------------
def _reference(db, uid, query_text, q, top_k=5, k1=1.2, b=0.75, rrf_k=60, cand=20):
    """Independent Python BM25 + RRF over the user's chunks (lexemes/positions from to_tsvector(content))."""
    rows = db.execute(text("SELECT c.id, c.content, c.embedding FROM document_chunks c JOIN documents d "
                           "ON d.id = c.document_id WHERE d.user_id = :u"), {"u": uid}).all()
    qlex = {r[0] for r in db.execute(text("SELECT unnest(tsvector_to_array(to_tsvector('english', :t)))"),
                                       {"t": query_text})}
    docs = {}
    for cid, content, emb in rows:
        lex = db.execute(text("SELECT lexeme, coalesce(array_length(positions, 1), 1) FROM "
                              "unnest(to_tsvector('english', :t))"), {"t": content}).all()
        emb = json.loads(emb) if isinstance(emb, str) else emb    # raw SQL returns pgvector's text form
        docs[cid] = ({k: v for k, v in lex}, sum(v for _, v in lex), [float(x) for x in emb])
    n = len(docs)
    avgdl = sum(d[1] for d in docs.values()) / n
    df = {t: sum(1 for d in docs.values() if t in d[0]) for t in qlex}
    bm25 = {}
    for cid, (tf, dl, _) in docs.items():
        terms = [t for t in qlex if t in tf]
        if terms:
            bm25[cid] = sum(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * tf[t] * (k1 + 1)
                            / (tf[t] + k1 * (1 - b + b * dl / avgdl)) for t in terms)

    def cos_dist(a):
        na, nq = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in q))
        return 1 - sum(x * y for x, y in zip(a, q)) / (na * nq)

    vec_rank = [c for c in sorted(docs, key=lambda c: (cos_dist(docs[c][2]), c))][:cand]
    fts_rank = sorted(bm25, key=lambda c: (-bm25[c], c))[:cand]
    ranks = {}
    for name, lst in (("v", vec_rank), ("f", fts_rank)):
        for r, c in enumerate(lst, 1):
            ranks.setdefault(c, {})[name] = r
    fused = sorted(ranks, key=lambda c: (-sum(1 / (rrf_k + r) for r in ranks[c].values()), min(ranks[c].values()), c))
    return fused[:top_k], bm25


def test_hybrid_matches_independent_bm25_rrf(db, make_library, hybrid):
    uid, _ = make_library(LIB)
    for question, q in (("How many robot trajectories was Octo trained on?", vec(1.0, 0.0)),
                        ("ALOHA kit cost and CVAE action chunking", vec(0.0, 1.0)),
                        ("robot", vec(0.3, 0.3, 0.3))):
        expected, _ = _reference(db, uid, question, q)
        got = R.retrieve_chunks(db, user_id=uid, query_text=question, query_embedding=q)
        assert [c.id for c in got] == expected, question
    assert R.fallback_counts() == {}


# --- isolation -----------------------------------------------------------------------------------------------
def test_other_users_never_appear_and_never_change_statistics(db, make_library, hybrid):
    uid, _ = make_library(LIB)
    question, q = "robot trajectories Open X-Embodiment", vec(1.0, 0.2)
    before = [(r.id, r.vector_rank, r.fts_rank, float(r.rrf_score)) for r in R.hybrid_ranking(db, uid, question, q)]
    other, _ = make_library({"spam.pdf": [("robot robot robot trajectories Open X-Embodiment " * 5, vec(1.0, 0.2))] * 4})
    after = [(r.id, r.vector_rank, r.fts_rank, float(r.rrf_score)) for r in R.hybrid_ranking(db, uid, question, q)]
    assert after == before                                         # scores and ranks unchanged
    other_ids = {c.id for c in db.query(DocumentChunk).join(Document).filter(Document.user_id == other)}
    mine = R.retrieve_chunks(db, user_id=uid, query_text=question, query_embedding=q, top_k=20)
    assert other_ids and not ({c.id for c in mine} & other_ids)
    theirs = R.retrieve_chunks(db, user_id=other, query_text=question, query_embedding=q, top_k=20)
    assert {c.id for c in theirs} <= other_ids


def test_document_filter_narrows_candidates_not_statistics(db, make_library, hybrid):
    uid, ids = make_library(LIB)
    question, q = "robot action", vec(1.0, 0.5)
    _, bm25_all = _reference(db, uid, question, q)
    rows = R.hybrid_ranking(db, uid, question, q, top_k=20, document_ids=[ids["octo.pdf"]])
    octo_chunks = {c.id for c in db.query(DocumentChunk).filter(DocumentChunk.document_id == ids["octo.pdf"])}
    assert {r.id for r in rows} <= octo_chunks
    fts_order = [r.id for r in sorted((r for r in rows if r.fts_rank), key=lambda r: r.fts_rank)]
    expected = sorted((c for c in bm25_all if c in octo_chunks), key=lambda c: (-bm25_all[c], c))
    assert fts_order == expected          # scores computed with whole-library statistics


# --- small libraries and degenerate queries ---------------------------------------------------------------
def test_empty_and_tiny_libraries(db, make_library, hybrid):
    empty, _ = make_library({})
    assert R.retrieve_chunks(db, user_id=empty, query_text="robot", query_embedding=vec(1.0)) == []
    tiny, _ = make_library({"one.pdf": [("A single chunk about robot grippers.", vec(1.0))]})
    got = R.retrieve_chunks(db, user_id=tiny, query_text="robot grippers", query_embedding=vec(1.0))
    assert len(got) == 1                                            # N = 1: IDF stays finite, no error
    two, _ = make_library({"two.pdf": [("grippers", vec(1.0)), ("unrelated text", vec(0.0, 1.0))]})
    got = R.retrieve_chunks(db, user_id=two, query_text="grippers", query_embedding=vec(0.0, 1.0))
    assert len(got) == 2 and R.fallback_counts() == {}


def test_query_without_lexemes_degrades_to_vector_order(db, make_library, hybrid, monkeypatch):
    uid, _ = make_library(LIB)
    q = vec(0.5, 0.5, 0.1)
    got = R.retrieve_chunks(db, user_id=uid, query_text="the of and", query_embedding=q)
    monkeypatch.setenv("RETRIEVAL_MODE", "vector")
    assert [c.id for c in got] == [c.id for c in R.retrieve_chunks(db, user_id=uid, query_text="x", query_embedding=q)]
    assert R.fallback_counts() == {"no_lexemes": 1}


# --- timeout / error fallback -----------------------------------------------------------------------------
@pytest.mark.parametrize("sql,timeout_ms,reason", [
    ("SELECT pg_sleep(2)", "50", "timeout"),
    ("SELECT * FROM table_that_does_not_exist", "300", "error"),
])
def test_fallback_recovers_the_transaction_and_logs_no_content(db, make_library, hybrid, monkeypatch, caplog,
                                                               sql, timeout_ms, reason):
    uid, _ = make_library(LIB)
    q = vec(1.0, 0.2)
    secret = "CONFIDENTIAL QUESTION TEXT"
    monkeypatch.setattr(R, "HYBRID_SQL", text(sql))
    monkeypatch.setenv("RETRIEVAL_HYBRID_TIMEOUT_MS", timeout_ms)
    original = db.execute(text("SHOW statement_timeout")).scalar()
    with caplog.at_level(logging.WARNING, logger="documind.retrieval"):
        got = R.retrieve_chunks(db, user_id=uid, query_text=secret, query_embedding=q)
    legacy = R.vector_chunks(db, uid, q)
    assert [c.id for c in got] == [c.id for c in legacy]           # vector fallback
    assert R.fallback_counts() == {reason: 1}
    assert db.execute(text("SHOW statement_timeout")).scalar() == original   # local timeout undone
    db.add(User(supabase_uid=str(uuid.uuid4())))                    # outer transaction still usable
    db.commit()
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert f"event=retrieval_fallback reason={reason} mode=hybrid_bm25" in messages
    assert secret not in messages and "robot" not in messages


def test_successful_hybrid_restores_statement_timeout(db, make_library, hybrid):
    uid, _ = make_library(LIB)
    original = db.execute(text("SHOW statement_timeout")).scalar()
    R.retrieve_chunks(db, user_id=uid, query_text="robot", query_embedding=vec(1.0))
    assert db.execute(text("SHOW statement_timeout")).scalar() == original


# --- lifecycle via the trigger --------------------------------------------------------------------------------
def _cols(db, chunk_id):
    return db.execute(text("SELECT content_tsv IS NOT NULL, token_count FROM document_chunks WHERE id = :i"),
                      {"i": chunk_id}).one()


def test_insert_update_delete_and_failed_ingestion(db, make_library, hybrid):
    uid, ids = make_library({"doc.pdf": [("robot grippers grasp objects", vec(1.0))]})
    chunk = db.query(DocumentChunk).filter(DocumentChunk.document_id == ids["doc.pdf"]).one()
    assert tuple(_cols(db, chunk.id)) == (True, 4)                  # filled on INSERT (upload / ingest)
    chunk.content = "robot robot robot grippers grasp objects firmly"
    db.commit()
    assert _cols(db, chunk.id)[1] == 7                              # recomputed on UPDATE of content
    # failed ingestion: chunks inserted then rolled back → nothing persists, statistics unchanged
    before = R.hybrid_ranking(db, uid, "robot", vec(1.0))
    doc = Document(user_id=uid, filename="failed.pdf", status="processing")
    db.add(doc)
    db.flush()
    db.add(DocumentChunk(document_id=doc.id, content="robot " * 50, embedding=vec(1.0)))
    db.flush()
    db.rollback()
    assert [tuple(r) for r in R.hybrid_ranking(db, uid, "robot", vec(1.0))] == [tuple(r) for r in before]
    # deletion via the API's ORM cascade → chunks gone, statistics change at commit
    db.delete(db.get(Document, ids["doc.pdf"]))
    db.commit()
    assert R.hybrid_ranking(db, uid, "robot", vec(1.0)) == []


# --- migration ------------------------------------------------------------------------------------------------------
def test_migration_up_down_up_idempotent_and_startup_validation(make_library, monkeypatch):
    conn = migration.connect(TEST_DATABASE_URL)
    try:
        assert migration.status(conn)["complete"]
        migration.up(conn, log=lambda *_: None)                     # idempotent
        migration.down(conn, log=lambda *_: None)
        st = migration.status(conn)
        assert st["columns"] == [] and not st["trigger"] and st["index"] == "missing"
        monkeypatch.setenv("RETRIEVAL_MODE", "hybrid_bm25")
        with pytest.raises(R.RetrievalConfigError, match="requires migration"):
            R.validate_configuration(engine)
        migration.up(conn, batch=2, log=lambda *_: None)           # batched backfill path
        assert migration.status(conn)["complete"] and R.validate_configuration(engine) == "hybrid_bm25"
        with conn.cursor() as cur:                                   # simulate a failed CONCURRENTLY build
            cur.execute("UPDATE pg_index SET indisvalid = false WHERE indexrelid = %s::regclass", (migration.INDEX,))
        assert migration.status(conn)["index"] == "invalid"
        migration.up(conn, log=lambda *_: None)
        assert migration.status(conn)["index"] == "valid"
    finally:
        conn.close()


def test_migration_refuses_non_local_databases():
    with pytest.raises(SystemExit, match="not approved"):
        migration.connect("postgresql://u:p@aws-0-eu.pooler.supabase.com:6543/postgres")


# --- end to end ----------------------------------------------------------------------------------------------------
def test_chat_endpoint_in_hybrid_mode(auth_client, hybrid, monkeypatch, db):
    import routers.chat as chat_router

    class FakeProvider:
        name = "fake"

        def embed(self, text, kind="document"):
            return vec(1.0, 0.2)

        async def stream_chat(self, system_prompt, history, query):
            assert "trajectories" in system_prompt                  # retrieved context reached the prompt
            yield "ok"

    monkeypatch.setattr(chat_router, "get_provider", lambda: FakeProvider())
    me = auth_client.get("/chat/sessions")                          # creates the local user row
    assert me.status_code == 200
    uid = db.query(User).order_by(User.id.desc()).first().id
    d = Document(user_id=uid, filename="octo.pdf", status="completed")
    db.add(d)
    db.flush()
    db.add(DocumentChunk(document_id=d.id, content=LIB["octo.pdf"][0][0], embedding=LIB["octo.pdf"][0][1]))
    db.commit()
    res = auth_client.post("/chat/chat", json={"query": "How many trajectories?"})
    assert res.status_code == 200 and res.text == "ok"
    assert R.fallback_counts() == {}
