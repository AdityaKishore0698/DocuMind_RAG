from unittest.mock import MagicMock, patch


@patch("routers.document.celery_app.send_task")
@patch("routers.document.get_supabase")
def test_upload_document(mock_get_supabase, mock_send_task, auth_client):
    # Storage upload and Celery enqueue are external — stub them out.
    mock_get_supabase.return_value = MagicMock()
    mock_send_task.return_value = MagicMock(id="test-task-id")

    response = auth_client.post(
        "/document/upload",
        files=[("files", ("test.txt", b"Hello, this is a test document.", "text/plain"))],
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["uploaded_files"]) == 1
    assert body["uploaded_files"][0]["filename"] == "test.txt"
    assert body["uploaded_files"][0]["task_id"] == "test-task-id"

    # The file bytes went to the "documents" bucket, and a task was enqueued.
    mock_get_supabase.return_value.storage.from_.assert_called_with("documents")
    mock_send_task.assert_called_once()


def test_list_documents_is_scoped_to_the_user(make_auth_client):
    a = make_auth_client(email="a@example.com")
    b = make_auth_client(email="b@example.com")
    # Neither user has uploaded anything; each sees an empty list, not an error.
    assert a.get("/document/").json() == []
    assert b.get("/document/").json() == []
