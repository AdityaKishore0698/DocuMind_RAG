def test_create_and_list_session(auth_client):
    res = auth_client.post("/chat/sessions", json={"title": "Test Session"})
    assert res.status_code == 200
    assert "id" in res.json()
    assert res.json()["title"] == "Test Session"

    res2 = auth_client.get("/chat/sessions")
    assert res2.status_code == 200
    assert len(res2.json()) == 1
    assert res2.json()[0]["title"] == "Test Session"


def test_rename_and_delete_session(auth_client):
    sid = auth_client.post("/chat/sessions", json={"title": "old"}).json()["id"]

    assert auth_client.put(f"/chat/sessions/{sid}", json={"title": "new"}).json()["title"] == "new"
    assert auth_client.delete(f"/chat/sessions/{sid}").status_code == 200
    assert auth_client.get("/chat/sessions").json() == []
