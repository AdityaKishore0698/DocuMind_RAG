from unittest.mock import MagicMock, patch


@patch("routers.document.celery_app.send_task")
@patch("routers.document.get_supabase")
def test_upload_document(mock_get_supabase, mock_send_task, auth_client):
    # Storage upload and Celery enqueue are external — stub them out.
    mock_get_supabase.return_value = MagicMock()
    mock_send_task.return_value = MagicMock(id="test-task-id")

    sid = auth_client.post("/chat/sessions", json={"title": "t"}).json()["id"]
    response = auth_client.post(
        "/document/upload",
        files=[("files", ("test.txt", b"Hello, this is a test document.", "text/plain"))],
        data={"session_id": sid},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["uploaded_files"]) == 1
    assert body["uploaded_files"][0]["filename"] == "test.txt"
    assert body["uploaded_files"][0]["task_id"] == "test-task-id"

    # The file bytes went to the "documents" bucket, and a task was enqueued.
    mock_get_supabase.return_value.storage.from_.assert_called_with("documents")
    mock_send_task.assert_called_once()
    assert mock_send_task.call_args.kwargs["args"][2] == sid          # the worker is told the paper's chat


def test_list_documents_is_scoped_to_the_user(make_auth_client):
    a = make_auth_client(email="a@example.com")
    b = make_auth_client(email="b@example.com")
    # Neither user has uploaded anything; each sees an empty list, not an error.
    sa = a.post("/chat/sessions", json={"title": "a"}).json()["id"]
    sb = b.post("/chat/sessions", json={"title": "b"}).json()["id"]
    assert a.get(f"/document/?session_id={sa}").json() == []
    assert b.get(f"/document/?session_id={sb}").json() == []
    assert a.get(f"/document/?session_id={sb}").status_code == 404     # not a's chat
