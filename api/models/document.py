from sqlalchemy import Column, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from core.database import Base

# Keep in sync with worker/models.py (worker/tests/test_ingest.py checks the shared columns).


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    # The chat the paper was uploaded in (migration 004). Papers are chat-scoped: retrieval, comparison and
    # gap analysis only ever see the current chat's papers. NULL = uploaded before papers belonged to chats;
    # such papers are searchable nowhere until the user explicitly moves each one into a chat.
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=True, index=True)
    filename = Column(String, index=True)
    status = Column(String, default="pending")
    # Paper metadata (migration 002). `doc_metadata` records where each field came from
    # (e.g. title_source = "pdf_metadata" | "first_page_heuristic") and detected sections.
    title = Column(String, nullable=True)
    authors = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    page_count = Column(Integer, nullable=True)
    doc_metadata = Column("metadata", JSONB, nullable=True)
    error = Column(Text, nullable=True)
    # Embedding space of this document's vectors, e.g. "ollama:nomic-embed-text" (migration 003).
    # Retrieval only searches documents whose space matches the running configuration.
    embedding_model = Column(String, nullable=True)

    user = relationship("User", back_populates="documents")
    session = relationship("ChatSession", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    # Re-ingesting a document replaces its chunks; this keeps (document, position) unique (migration 002).
    __table_args__ = (Index("uq_document_chunks_document_chunk_index", "document_id", "chunk_index", unique=True),)

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    content = Column(String)
    embedding = Column(Vector(768))  # Gemini gemini-embedding-2 (output_dimensionality=768)
    # Provenance (migration 002): page is null for plain-text uploads; section when detected.
    page_number = Column(Integer, nullable=True)
    section_heading = Column(String, nullable=True)
    chunk_index = Column(Integer, nullable=True)

    document = relationship("Document", back_populates="chunks")
