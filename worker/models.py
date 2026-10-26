from sqlalchemy import Column, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from database import Base

# Keep in sync with api/models/document.py (worker/tests/test_ingest.py checks the shared columns).


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    # Chat the paper belongs to (migration 004); ingestion never changes it.
    session_id = Column(Integer, nullable=True, index=True)
    filename = Column(String, index=True)
    status = Column(String, default="pending")  # uploaded, processing, completed, failed
    title = Column(String, nullable=True)
    authors = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    page_count = Column(Integer, nullable=True)
    doc_metadata = Column("metadata", JSONB, nullable=True)
    error = Column(Text, nullable=True)
    # Embedding space of this document's vectors, e.g. "ollama:nomic-embed-text" (migration 003).
    # Retrieval only searches documents whose space matches the running configuration.
    embedding_model = Column(String, nullable=True)

    chunks = relationship("DocumentChunk", back_populates="document")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    # Re-ingesting a document replaces its chunks; this keeps (document, position) unique (migration 002).
    __table_args__ = (Index("uq_document_chunks_document_chunk_index", "document_id", "chunk_index", unique=True),)

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    content = Column(Text)
    embedding = Column(Vector(768))  # Gemini gemini-embedding-2 (output_dimensionality=768)
    page_number = Column(Integer, nullable=True)
    section_heading = Column(String, nullable=True)
    chunk_index = Column(Integer, nullable=True)

    document = relationship("Document", back_populates="chunks")
