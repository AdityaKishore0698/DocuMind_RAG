import os
from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from celery import Celery
from sqlalchemy.orm import Session
from core.database import get_db
from core.storage import get_supabase, object_key, STORAGE_BUCKET
from models.document import Document
from models.user import User
from core.dependencies import get_current_user

router = APIRouter()


def _redis_url() -> str:
    # kombu needs an explicit ssl_cert_reqs on rediss:// (TLS) broker URLs.
    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    if url.startswith("rediss://") and "ssl_cert_reqs" not in url:
        url += ("&" if "?" in url else "?") + "ssl_cert_reqs=required"
    return url


celery_app = Celery("worker", broker=_redis_url())

@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    supabase = get_supabase()
    results = []

    for file in files:
        file_bytes = await file.read()

        db_document = Document(filename=file.filename, status="uploaded", user_id=current_user.id)
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

        # Hand the Storage path (not a disk path) to the worker.
        task = celery_app.send_task("tasks.process_document", args=[db_document.id, storage_path])
        results.append({"task_id": task.id, "document_id": db_document.id, "filename": file.filename})

    return {"uploaded_files": results}

@router.get("/")
def list_documents(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    docs = db.query(Document).filter(Document.user_id == current_user.id).all()
    return [{"id": d.id, "filename": d.filename, "status": d.status} for d in docs]

@router.delete("/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    doc = db.query(Document).filter(Document.id == document_id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove the object from Supabase Storage (best effort — don't block the DB delete).
    try:
        get_supabase().storage.from_(STORAGE_BUCKET).remove(
            [object_key(doc.user_id, doc.id, doc.filename)]
        )
    except Exception:
        pass

    db.delete(doc)
    db.commit()
    return {"message": "Document deleted"}
