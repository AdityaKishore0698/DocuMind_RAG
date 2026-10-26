import os
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from celery import Celery
from sqlalchemy.orm import Session
from core.database import get_db
from core.storage import get_supabase, object_key, remove_stored_files, STORAGE_BUCKET
from models.document import Document
from models.user import User
from core.dependencies import get_current_user
from core.inference import embedding_space
from core.ownership import owned_session

router = APIRouter()


def _redis_url() -> str:
    # kombu needs an explicit ssl_cert_reqs on rediss:// (TLS) broker URLs.
    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    if url.startswith("rediss://") and "ssl_cert_reqs" not in url:
        url += ("&" if "?" in url else "?") + "ssl_cert_reqs=required"
    return url


celery_app = Celery("worker", broker=_redis_url())

ALLOWED_EXTENSIONS = {".pdf": "application/pdf", ".txt": "text/plain"}


def max_upload_bytes() -> int:
    return int(float(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024)


def _check_upload(filename: str, data: bytes) -> None:
    """Reject unsupported or oversized files before anything is stored or queued."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"{filename}: only PDF and TXT files are supported")
    if not data:
        raise HTTPException(status_code=400, detail=f"{filename}: the file is empty")
    if len(data) > max_upload_bytes():
        raise HTTPException(status_code=413, detail=f"{filename}: larger than the {os.getenv('MAX_UPLOAD_MB', '20')} MB limit")
    if ext == ".pdf" and not data.startswith(b"%PDF"):
        raise HTTPException(status_code=415, detail=f"{filename}: not a valid PDF file")


def document_summary(d: Document, space: str) -> dict:
    return {"id": d.id, "session_id": d.session_id, "filename": d.filename, "status": d.status, "title": d.title,
            "authors": d.authors,
            "year": d.year, "page_count": d.page_count, "error": d.error,
            # False for a paper embedded with another model: it is excluded from search until re-processed.
            "embedding_model": d.embedding_model,
            "searchable": d.status == "completed" and d.embedding_model == space,
            "sections": (d.doc_metadata or {}).get("sections", []),
            "metadata_sources": {k: v for k, v in (d.doc_metadata or {}).items() if k.endswith("_source")}}

def _chat_document(db: Session, user: User, document_id: int, session_id: int) -> Document:
    """A paper of this user in this chat, else 404 (same answer for another user's or another chat's paper)."""
    doc = db.query(Document).filter(Document.id == document_id, Document.user_id == user.id,
                                    Document.session_id == session_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: int = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Upload papers into chat `session_id` (papers are chat-scoped; the chat must belong to the caller)."""
    chat = owned_session(db, current_user.id, session_id)
    payloads = []
    for file in files:  # validate every file first: a bad file rejects the whole request
        file_bytes = await file.read()
        _check_upload(file.filename, file_bytes)
        payloads.append((file, file_bytes))

    supabase = get_supabase()
    results = []
    for file, file_bytes in payloads:

        db_document = Document(filename=file.filename, status="uploaded", user_id=current_user.id, session_id=chat.id)
        db.add(db_document)
        db.commit()
        db.refresh(db_document)

        # Upload the bytes straight to Supabase Storage — no local disk.
        storage_path = object_key(current_user.id, db_document.id, file.filename)
        supabase.storage.from_(STORAGE_BUCKET).upload(
            storage_path,
            file_bytes,
            {
                "content-type": file.content_type or "application/octet-stream",
                "upsert": "true",
            },
        )

        # Hand the Storage path (not a disk path) and the expected chat to the worker.
        task = celery_app.send_task("tasks.process_document", args=[db_document.id, storage_path, chat.id])
        results.append({"task_id": task.id, "document_id": db_document.id, "filename": file.filename})

    return {"uploaded_files": results}

@router.get("/")
def list_documents(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """The papers of chat `session_id` only."""
    owned_session(db, current_user.id, session_id)
    docs = db.query(Document).filter(Document.user_id == current_user.id, Document.session_id == session_id) \
        .order_by(Document.id).all()
    space = embedding_space()
    return [document_summary(d, space) for d in docs]


@router.get("/unassigned")
def list_unassigned_documents(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Papers uploaded before papers belonged to chats (migration 004). They are searched in no chat until the user
    moves one into a chat; they are never assigned automatically."""
    docs = db.query(Document).filter(Document.user_id == current_user.id, Document.session_id.is_(None)) \
        .order_by(Document.id).all()
    space = embedding_space()
    return [document_summary(d, space) for d in docs]


@router.post("/{document_id}/assign")
def assign_document(document_id: int, session_id: int, db: Session = Depends(get_db),
                    current_user: User = Depends(get_current_user)):
    """Move one unassigned legacy paper into chat `session_id`. A paper that already belongs to a chat is never
    moved or shared (404, like a missing one)."""
    chat = owned_session(db, current_user.id, session_id)
    doc = db.query(Document).filter(Document.id == document_id, Document.user_id == current_user.id,
                                    Document.session_id.is_(None)).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    doc.session_id = chat.id
    db.commit()
    return document_summary(doc, embedding_space())


@router.post("/{document_id}/reprocess")
def reprocess_document(document_id: int, session_id: int, db: Session = Depends(get_db),
                       current_user: User = Depends(get_current_user)):
    """Re-ingest a stored paper of chat `session_id` with the current embedding model (e.g. after switching
    providers). It stays in its chat; ingestion replaces its chunks in one transaction, so a repeated request (or a
    document stuck in `processing` after a worker crash) is safe to re-run."""
    doc = _chat_document(db, current_user, document_id, session_id)
    doc.status, doc.error = "uploaded", None
    db.commit()
    task = celery_app.send_task("tasks.process_document",
                                args=[doc.id, object_key(current_user.id, doc.id, doc.filename), doc.session_id])
    return {"task_id": task.id, "document_id": doc.id}

@router.delete("/{document_id}")
def delete_document(document_id: int, session_id: Optional[int] = None, db: Session = Depends(get_db),
                    current_user: User = Depends(get_current_user)):
    """Delete a paper of chat `session_id`; without `session_id`, only an unassigned legacy paper."""
    if session_id is None:
        doc = db.query(Document).filter(Document.id == document_id, Document.user_id == current_user.id,
                                        Document.session_id.is_(None)).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
    else:
        doc = _chat_document(db, current_user, document_id, session_id)

    key = object_key(doc.user_id, doc.id, doc.filename)
    db.delete(doc)
    db.commit()
    # The stored file goes after the row is committed; a failure is reported, not swallowed.
    return {"message": "Document deleted", "files_not_removed": remove_stored_files([key])}
