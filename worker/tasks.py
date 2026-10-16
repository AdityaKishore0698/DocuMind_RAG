import io
import os
import PyPDF2
from celery_app import celery_app
from database import SessionLocal
from models import Document, DocumentChunk
from storage import get_supabase, STORAGE_BUCKET
from langchain_text_splitters import RecursiveCharacterTextSplitter
from inference import get_embedding  # Gemini by default; INFERENCE_PROVIDER=ollama -> nomic-embed-text

def extract_text(data: bytes, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        text = ""
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    return data.decode("utf-8", errors="replace")

@celery_app.task
def process_document(document_id: int, storage_path: str):
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return f"Document {document_id} not found"

        doc.status = "processing"
        db.commit()

        # Pull the file bytes from Supabase Storage into memory.
        try:
            file_bytes = get_supabase().storage.from_(STORAGE_BUCKET).download(storage_path)
        except Exception as e:
            doc.status = "failed"
            db.commit()
            return f"Could not download {storage_path}: {e}"

        text = extract_text(file_bytes, doc.filename)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)

        for chunk_text in chunks:
            embedding = get_embedding(chunk_text)

            chunk_record = DocumentChunk(
                document_id=document_id,
                content=chunk_text,
                embedding=embedding
            )
            db.add(chunk_record)

        doc.status = "completed"
        db.commit()

        return f"Document {document_id} processed successfully"
    except Exception as e:
        db.rollback()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "failed"
            db.commit()
        raise e
    finally:
        db.close()
