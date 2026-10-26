"""Celery tasks: page/section-aware document ingestion (plan §7.1).

process_document: download → extract pages → chunk within page/section boundaries → embed in batches →
replace the document's chunks and metadata in ONE transaction → status `completed`. Re-running the task
(a retry or a re-ingest) replaces rather than duplicates chunks. Any failure leaves no partial chunks and
records status `failed` with a short, content-free reason in `documents.error`.
"""

import logging

from celery_app import celery_app
from database import SessionLocal
from ingest import chunk_pages, extract_pages, paper_metadata
from models import Document, DocumentChunk
from storage import STORAGE_BUCKET, get_supabase
from inference import EmbeddingQuotaExhausted, embedding_space, get_embeddings  # INFERENCE_PROVIDER selects

log = logging.getLogger("documind.worker")


class IngestionError(RuntimeError):
    """A user-facing, content-free reason for a failed ingestion."""


def _fail(db, document_id: int, reason: str) -> str:
    db.rollback()
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc:
        doc.status, doc.error = "failed", reason[:500]
        db.commit()
    log.warning("event=ingest_failed document_id=%s reason=%s", document_id, reason)
    return f"Document {document_id} failed: {reason}"


def ingest_bytes(db, doc: Document, data: bytes, embed=get_embeddings) -> int:
    """Extract, chunk, embed and store; returns the number of chunks. Raises IngestionError."""
    try:
        extracted = extract_pages(data, doc.filename)
    except Exception as exc:  # noqa: BLE001 — malformed PDFs
        raise IngestionError(f"could not read the file ({type(exc).__name__})") from exc
    chunks = chunk_pages(extracted.pages)
    if not chunks:
        raise IngestionError("no extractable text (scanned or image-only PDF?)")
    vectors = embed([c.content for c in chunks])
    if len(vectors) != len(chunks):
        raise IngestionError("embedding count mismatch")
    meta = paper_metadata(extracted, chunks)
    # One transaction: drop any previous chunks of this document, insert the new ones, update metadata.
    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete(synchronize_session=False)
    db.add_all(DocumentChunk(document_id=doc.id, content=c.content, embedding=v, page_number=c.page_number,
                             section_heading=c.section_heading, chunk_index=c.chunk_index)
               for c, v in zip(chunks, vectors))
    doc.title = meta.get("title")
    doc.authors = meta.get("authors")
    doc.year = meta.get("year")
    doc.page_count = sum(1 for p in extracted.pages if p.page_number is not None) or None
    doc.doc_metadata = meta
    doc.embedding_model = embedding_space()  # the space these vectors live in (never mixed at query time)
    doc.status, doc.error = "completed", None
    db.commit()
    return len(chunks)


@celery_app.task
def process_document(document_id: int, storage_path: str, session_id: int | None = None):
    """`session_id` is the chat the API enqueued the paper for. Ingestion never changes a paper's chat; a task whose
    chat does not match the row (a defensive check against stale or forged tasks) is dropped, not ingested.
    Two-argument tasks queued before chat-scoped papers still work."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return f"Document {document_id} not found"
        if session_id is not None and doc.session_id != session_id:
            log.warning("event=ingest_skipped document_id=%s reason=chat_mismatch", document_id)
            return f"Document {document_id} skipped: it no longer belongs to chat {session_id}"
        doc.status, doc.error = "processing", None
        db.commit()
        try:
            data = get_supabase().storage.from_(STORAGE_BUCKET).download(storage_path)
        except Exception as exc:  # noqa: BLE001
            return _fail(db, document_id, f"could not download the file ({type(exc).__name__})")
        try:
            count = ingest_bytes(db, doc, data)
        except IngestionError as exc:
            return _fail(db, document_id, str(exc))
        except EmbeddingQuotaExhausted:
            return _fail(db, document_id, "embedding quota exhausted for today — re-upload later")
        except Exception as exc:  # noqa: BLE001 — embedding/provider/database errors
            return _fail(db, document_id, f"processing error ({type(exc).__name__})")
        log.info("event=ingest_completed document_id=%s chunks=%s", document_id, count)
        return f"Document {document_id} processed successfully ({count} chunks)"
    finally:
        db.close()
