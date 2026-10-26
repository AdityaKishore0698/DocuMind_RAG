"""Chat-scoped papers: a paper belongs to the chat it was uploaded in. It is listed, retrieved, compared and analysed
only there; ids of another chat's or another user's papers are rejected (404) everywhere."""

import pytest
from unittest.mock import MagicMock

import core.agent as agent_core
import routers.chat as chat_router
import routers.document as doc_router
from models.document import Document, DocumentChunk
from models.user import ChatSession, User
from test_research_features import SPACE, FakeProvider, _sources, vec


@pytest.fixture
def db(db_session):
    return db_session


@pytest.fixture
def fake(monkeypatch):
    import routers.analysis as analysis_router

    p = FakeProvider(json_replies=[None] * 10)
    monkeypatch.setattr(chat_router, "get_provider", lambda: p)
    monkeypatch.setattr(analysis_router, "get_provider", lambda: p)
    return p


@pytest.fixture
def storage(monkeypatch):
    sent, supa = [], MagicMock()
    monkeypatch.setattr(doc_router, "get_supabase", lambda: supa)
    import core.storage as storage_module

    monkeypatch.setattr(storage_module, "get_supabase", lambda: supa)
    monkeypatch.setattr(doc_router.celery_app, "send_task",
                        lambda name, args: sent.append((name, args)) or MagicMock(id=f"t{len(sent)}"))
    return {"sent": sent, "supabase": supa}


def ingest(db, doc_id, text, emb):
    """What the worker would do (tests have no worker): store one chunk and mark the paper ready."""
    db.add(DocumentChunk(document_id=doc_id, content=text, embedding=emb, page_number=1, chunk_index=0))
    doc = db.get(Document, doc_id)
    doc.status, doc.embedding_model = "completed", SPACE
    db.commit()


@pytest.fixture
def two_chats(make_auth_client, db, storage):
    """User `me` with Chat A (paper "alpha.txt") and Chat B (paper "beta.txt"), uploaded through the API."""
    me = make_auth_client(email="scope@example.com")
    a = me.post("/chat/sessions", json={"title": "Chat A"}).json()["id"]
    b = me.post("/chat/sessions", json={"title": "Chat B"}).json()["id"]
    up_a = me.post("/document/upload", files=[("files", ("alpha.txt", b"alpha text"))], data={"session_id": a})
    up_b = me.post("/document/upload", files=[("files", ("beta.txt", b"beta text"))], data={"session_id": b})
    doc_a, doc_b = up_a.json()["uploaded_files"][0]["document_id"], up_b.json()["uploaded_files"][0]["document_id"]
    ingest(db, doc_a, "ALPHA: the answer is in chat A.", vec(1.0, 0.1))
    ingest(db, doc_b, "BETA: chat B content.", vec(0.9, 0.2))
    return {"me": me, "a": a, "b": b, "doc_a": doc_a, "doc_b": doc_b, **storage}


def test_upload_records_the_chat_and_tells_the_worker(two_chats, db):
    assert db.get(Document, two_chats["doc_a"]).session_id == two_chats["a"]
    assert [args[2] for _, args in two_chats["sent"]] == [two_chats["a"], two_chats["b"]]


def test_each_chat_lists_only_its_own_papers(two_chats):
    me = two_chats["me"]
    assert [d["filename"] for d in me.get(f"/document/?session_id={two_chats['a']}").json()] == ["alpha.txt"]
    assert [d["filename"] for d in me.get(f"/document/?session_id={two_chats['b']}").json()] == ["beta.txt"]
    sessions = {s["id"]: s["paper_count"] for s in me.get("/chat/sessions").json()}
    assert sessions[two_chats["a"]] == 1 and sessions[two_chats["b"]] == 1


def test_a_paper_is_retrieved_in_its_chat_and_never_in_another(two_chats, fake):
    me = two_chats["me"]
    in_a = me.post("/chat/chat", json={"query": "answer?", "session_id": two_chats["a"]})
    assert {s["document_id"] for s in _sources(in_a)} == {two_chats["doc_a"]}
    assert "ALPHA" in fake.turns[-1] and "BETA" not in fake.turns[-1]
    in_b = me.post("/chat/chat", json={"query": "answer?", "session_id": two_chats["b"]})
    assert {s["document_id"] for s in _sources(in_b)} == {two_chats["doc_b"]}
    assert "ALPHA" not in fake.turns[-1]


def test_a_new_chat_sees_no_earlier_papers(two_chats, fake):
    res = two_chats["me"].post("/chat/chat", json={"query": "answer?"})
    assert res.text == chat_router.NO_PAPERS and _sources(res) == [] and fake.stream_calls == 0


def test_another_chats_paper_ids_are_rejected_everywhere(two_chats, fake):
    me, b, doc_a = two_chats["me"], two_chats["b"], two_chats["doc_a"]
    assert me.post("/chat/chat", json={"query": "q", "session_id": b, "document_ids": [doc_a]}).status_code == 404
    for path in ("/analysis/compare", "/analysis/gaps"):
        res = me.post(path, json={"session_id": b, "document_ids": [two_chats["doc_b"], doc_a]})
        assert res.status_code == 404
    assert me.post(f"/document/{doc_a}/reprocess?session_id={b}").status_code == 404
    assert me.delete(f"/document/{doc_a}?session_id={b}").status_code == 404
    assert me.delete(f"/document/{doc_a}").status_code == 404     # without a chat only unassigned papers qualify
    # the paper is untouched and still works in its own chat
    assert [d["id"] for d in me.get(f"/document/?session_id={two_chats['a']}").json()] == [doc_a]


def test_selection_without_a_chat_is_rejected(two_chats, fake):
    res = two_chats["me"].post("/chat/chat", json={"query": "q", "document_ids": [two_chats["doc_a"]]})
    assert res.status_code == 404


def test_another_user_cannot_reach_the_chat_or_its_papers(two_chats, make_auth_client, fake, storage):
    intruder = make_auth_client(email="intruder@example.com")
    a, doc_a = two_chats["a"], two_chats["doc_a"]
    own = intruder.post("/chat/sessions", json={"title": "mine"}).json()["id"]
    assert intruder.get(f"/document/?session_id={a}").status_code == 404
    assert intruder.get(f"/chat/sessions/{a}/messages").status_code == 404
    assert intruder.post("/chat/chat", json={"query": "q", "session_id": a}).status_code == 404
    assert intruder.post("/chat/chat", json={"query": "q", "session_id": own, "document_ids": [doc_a]}).status_code == 404
    assert intruder.post("/analysis/compare", json={"session_id": a, "document_ids": [doc_a, two_chats["doc_b"]]}) \
        .status_code == 404
    assert intruder.post(f"/document/{doc_a}/reprocess?session_id={a}").status_code == 404
    assert intruder.delete(f"/document/{doc_a}?session_id={a}").status_code == 404
    assert intruder.post(f"/document/{doc_a}/assign?session_id={own}").status_code == 404
    sent_before = len(storage["sent"])
    res = intruder.post("/document/upload", files=[("files", ("x.txt", b"x"))], data={"session_id": a})
    assert res.status_code == 404 and len(storage["sent"]) == sent_before


def test_upload_needs_a_chat(two_chats):
    res = two_chats["me"].post("/document/upload", files=[("files", ("x.txt", b"x"))])
    assert res.status_code == 422


def test_reprocess_keeps_the_chat(two_chats, db):
    me, a, doc_a = two_chats["me"], two_chats["a"], two_chats["doc_a"]
    assert me.post(f"/document/{doc_a}/reprocess?session_id={a}").status_code == 200
    assert two_chats["sent"][-1][1][2] == a
    db.expire_all()
    assert db.get(Document, doc_a).session_id == a


def test_deep_analysis_only_sees_the_chat_papers(two_chats, fake, monkeypatch):
    monkeypatch.setenv("AGENT_ENABLED", "true")
    fake.json_replies = [{"subqueries": ["alpha", "beta"]}, {"sufficient": True, "followup_query": ""}]
    res = two_chats["me"].post("/chat/chat", json={"query": "everything", "session_id": two_chats["b"],
                                                   "mode": "agent"})
    assert res.headers["X-Chat-Mode"] == "agent"
    assert {s["document_id"] for s in _sources(res)} == {two_chats["doc_b"]}


def test_deleting_a_chat_deletes_its_papers_and_files_only(two_chats, db):
    me, a, doc_a, doc_b = two_chats["me"], two_chats["a"], two_chats["doc_a"], two_chats["doc_b"]
    res = me.delete(f"/chat/sessions/{a}")
    assert res.status_code == 200 and res.json()["papers_deleted"] == 1 and res.json()["files_not_removed"] == 0
    removed = two_chats["supabase"].storage.from_.return_value.remove.call_args.args[0]
    uid = db.get(Document, doc_b).user_id
    assert removed == [f"{uid}/{doc_a}/alpha.txt"]
    db.expire_all()
    assert db.get(Document, doc_a) is None
    assert db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_a).count() == 0
    assert db.get(Document, doc_b) is not None


def test_a_chat_created_by_an_upload_is_named_by_its_first_question(two_chats, fake, storage):
    me = two_chats["me"]
    sid = me.post("/chat/sessions", json={"title": chat_router.NEW_CHAT_TITLE}).json()["id"]
    me.post("/chat/chat", json={"query": "What is in the new chat?", "session_id": sid})
    titles = {s["id"]: s["title"] for s in me.get("/chat/sessions").json()}
    assert titles[sid] == "What is in the new chat?"[:30]


# --- papers uploaded before chat scoping (migration 004: session_id NULL) ---------------------------------
@pytest.fixture
def legacy(two_chats, db):
    uid = db.get(Document, two_chats["doc_a"]).user_id
    d = Document(user_id=uid, filename="legacy.pdf", title="Legacy", status="completed", embedding_model=SPACE)
    db.add(d)
    db.flush()
    db.add(DocumentChunk(document_id=d.id, content="LEGACY content.", embedding=vec(1.0, 0.1), chunk_index=0))
    db.commit()
    return d.id


def test_unassigned_papers_are_searched_in_no_chat(two_chats, legacy, fake):
    me = two_chats["me"]
    for sid in (two_chats["a"], two_chats["b"]):
        res = me.post("/chat/chat", json={"query": "legacy?", "session_id": sid})
        assert legacy not in {s["document_id"] for s in _sources(res)}
        assert "LEGACY" not in fake.turns[-1]
    assert me.post("/chat/chat", json={"query": "q", "session_id": two_chats["a"], "document_ids": [legacy]}) \
        .status_code == 404
    assert [d["id"] for d in me.get("/document/unassigned").json()] == [legacy]


def test_an_unassigned_paper_can_be_moved_into_exactly_one_chat(two_chats, legacy, fake, make_auth_client):
    me, a, b = two_chats["me"], two_chats["a"], two_chats["b"]
    assert make_auth_client(email="x@example.com").post(f"/document/{legacy}/assign?session_id={a}").status_code == 404
    res = me.post(f"/document/{legacy}/assign?session_id={a}")
    assert res.status_code == 200 and res.json()["session_id"] == a
    assert me.get("/document/unassigned").json() == []
    assert me.post(f"/document/{legacy}/assign?session_id={b}").status_code == 404     # never moved or shared again
    in_a = me.post("/chat/chat", json={"query": "legacy?", "session_id": a, "document_ids": [legacy]})
    assert {s["document_id"] for s in _sources(in_a)} == {legacy}


def test_unassigned_paper_can_be_deleted_without_a_chat(two_chats, legacy, db):
    assert two_chats["me"].delete(f"/document/{legacy}").status_code == 200
    db.expire_all()
    assert db.get(Document, legacy) is None


def test_stored_files_are_removed_only_after_the_rows_are_gone(two_chats, db, monkeypatch):
    import core.storage as storage_module
    from conftest import TestingSessionLocal

    seen = {}

    def remove(keys):
        probe = TestingSessionLocal()   # a separate connection sees only committed state
        seen["rows_left"] = probe.query(Document).filter(Document.id == two_chats["doc_a"]).count()
        probe.close()
        seen["keys"] = keys

    two_chats["supabase"].storage.from_.return_value.remove.side_effect = remove
    assert two_chats["me"].delete(f"/chat/sessions/{two_chats['a']}").status_code == 200
    assert seen["rows_left"] == 0 and len(seen["keys"]) == 1


def test_a_storage_failure_is_reported_not_swallowed(two_chats, db):
    two_chats["supabase"].storage.from_.return_value.remove.side_effect = RuntimeError("storage down")
    res = two_chats["me"].delete(f"/chat/sessions/{two_chats['a']}")
    assert res.status_code == 200 and res.json()["files_not_removed"] == 1   # the client can warn the user
    db.expire_all()
    assert db.get(Document, two_chats["doc_a"]) is None                         # the chat and papers are gone
    res = two_chats["me"].delete(f"/document/{two_chats['doc_b']}?session_id={two_chats['b']}")
    assert res.status_code == 200 and res.json()["files_not_removed"] == 1


def test_deleting_another_users_chat_is_a_404_and_deletes_nothing(two_chats, make_auth_client, db):
    intruder = make_auth_client(email="deleter@example.com")
    assert intruder.delete(f"/chat/sessions/{two_chats['a']}").status_code == 404
    db.expire_all()
    assert db.get(Document, two_chats["doc_a"]) is not None
    two_chats["supabase"].storage.from_.return_value.remove.assert_not_called()
