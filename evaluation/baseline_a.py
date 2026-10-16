"""System A — a frozen reproduction of DocuMind's production RAG pipeline.

Everything here is copied from the baseline commit ``BASELINE_COMMIT`` so the
baseline stays fixed even after Phases 2-6 change ``worker/tasks.py`` and
``api/routers/chat.py``. ``tests/test_fidelity.py`` re-reads both files *at that
commit* (``git show``) and asserts that every constant, the prompt and the
``extract_text`` body below are identical to production.

Production behaviour reproduced (source line references are at the baseline commit):

* extraction  — worker/tasks.py:24-34   ``extract_text`` (PyPDF2, pages joined by "\\n")
* chunking    — worker/tasks.py:65      ``RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)``
* embeddings  — worker/tasks.py:36-42 / api/routers/chat.py:39-45
                ``gemini-embedding-2``, ``EmbedContentConfig(output_dimensionality=768)``, no task type
* retrieval   — api/routers/chat.py:107-111 cosine distance (pgvector ``<=>``), top 5,
                filtered by ``documents.user_id``; no document filter, no tie-breaker
* context     — api/routers/chat.py:113 ``"\\n\\n".join(chunk.content ...)`` in rank order
* prompt      — api/routers/chat.py:118-123 (``system_instruction``)
* generation  — api/routers/chat.py:138-142 ``client.aio.models.generate_content_stream``
                with ``gemini-3.6-flash``, no temperature / thinking config. A benchmark
                question is a fresh session, so ``contents`` holds only the user turn.

Deliberate, documented differences (do not affect what System A retrieves or answers):

* Ingestion embeddings may be *batched* (one request carrying many ``Content``
  objects) to save quota; ``run_eval.py check-embedding-equivalence`` verifies
  that batched vectors match the per-chunk production call. ``--embed-mode
  single`` reproduces the exact per-chunk call pattern.
* Chunk -> page provenance is computed *by the harness* (provenance.py) purely
  for scoring; System A itself never sees page numbers.

Evaluation-only preprocessing (deliberately NOT in this file):
``preprocess.strip_nul`` removes NUL characters after extraction, because
PostgreSQL cannot store them. It is applied by the System A entry in
run_eval.py, recorded in every run's metadata, and documented in
preprocess.py. This module stays a verbatim copy of production.
"""

from __future__ import annotations

import io
import os

import PyPDF2  # noqa: F401  (used by the frozen extract_text below)

BASELINE_COMMIT = "4a89dc18b72bac98d639915ad5893570c55bc770"  # DocuMind_RAG "Initial commit"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 768
CHAT_MODEL = "gemini-3.6-flash"
TOP_K = 5

# api/routers/chat.py:118-123 — the f-string's only placeholder is {context}.
SYSTEM_PROMPT_TEMPLATE = (
    "You are DocuMind, an assistant that answers questions using the user's "
    "uploaded documents. Rely on the context below; if it does not contain the "
    "answer, say so.\n\n"
    "Context from documents:\n{context}"
)

# Equivalent of the SQLAlchemy query in api/routers/chat.py:107-111:
#   db.query(DocumentChunk).join(Document)
#     .filter(Document.user_id == current_user.id)
#     .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
#     .limit(5)
RETRIEVAL_SQL = """
SELECT document_chunks.id, document_chunks.document_id, document_chunks.content,
       document_chunks.embedding <=> %(query_embedding)s AS cosine_distance
FROM document_chunks
JOIN documents ON documents.id = document_chunks.document_id
WHERE documents.user_id = %(user_id)s
ORDER BY document_chunks.embedding <=> %(query_embedding)s
LIMIT %(top_k)s
"""


# --- verbatim copy of worker/tasks.py:24-34 at BASELINE_COMMIT ----------------
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
# -----------------------------------------------------------------------------


def make_splitter():
    """The production splitter (worker/tasks.py:65)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def split_text(text: str) -> list[str]:
    return make_splitter().split_text(text)


def build_context(chunk_contents: list[str]) -> str:
    """api/routers/chat.py:113."""
    return "\n\n".join(chunk_contents)


def build_system_prompt(context: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(context=context)
