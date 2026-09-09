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
    assert c.get("/document/").status_code == 200
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
