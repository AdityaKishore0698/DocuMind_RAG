import uuid


def test_missing_token_is_rejected(client):
    assert client.get("/document/").status_code in (401, 403)
    assert client.get("/chat/sessions").status_code in (401, 403)


def test_garbage_token_is_rejected(client):
    r = client.get("/document/", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


def test_first_request_upserts_a_profile(make_auth_client):
    sub = str(uuid.uuid4())
    c = make_auth_client(sub=sub, email="alice@example.com")

    # First authenticated call creates the local users row...
    assert c.get("/document/unassigned").status_code == 200
    assert c.get("/chat/sessions").json() == []

    # ...and a second call reuses it (no duplicate, no error).
    r = c.post("/chat/sessions", json={"title": "s1"})
    assert r.status_code == 200
    assert c.get("/chat/sessions").json()[0]["title"] == "s1"


def test_users_are_isolated(make_auth_client):
    a = make_auth_client(email="a@example.com")
    b = make_auth_client(email="b@example.com")

    a.post("/chat/sessions", json={"title": "a-session"})

    assert [s["title"] for s in a.get("/chat/sessions").json()] == ["a-session"]
    assert b.get("/chat/sessions").json() == []


def test_profile_upsert_survives_a_concurrent_first_request(db_session, monkeypatch):
    """A new user's first page load fires several requests at once. If another request creates the profile
    between our lookup and our insert, we must return that row instead of failing with a 500."""
    from fastapi.security import HTTPAuthorizationCredentials

    import core.dependencies as deps
    from conftest import TestingSessionLocal
    from models.user import User

    monkeypatch.setattr(deps, "verify_supabase_jwt", lambda token: {"sub": "race-uid", "email": "race@example.com"})
    real_add = db_session.add

    def add_after_a_competing_insert(obj):
        other = TestingSessionLocal()
        other.add(User(supabase_uid="race-uid", email="race@example.com"))  # the other request wins the race
        other.commit()
        other.close()
        real_add(obj)

    monkeypatch.setattr(db_session, "add", add_after_a_competing_insert)
    user = deps.get_current_user(HTTPAuthorizationCredentials(scheme="Bearer", credentials="t"), db_session)
    assert user.supabase_uid == "race-uid"
    assert db_session.query(User).filter(User.supabase_uid == "race-uid").count() == 1
