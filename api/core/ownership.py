"""Ownership validation: chats belong to users, papers belong to chats.

Every endpoint that touches a chat or its papers resolves the chat with `owned_session` (user check) and the
selected papers with `owned_document_ids` (user + chat check). An id belonging to another user or another chat
gets the same 404 as a missing one, so changing an id in a request reveals nothing.
"""

from __future__ import annotations

from typing import Optional, Sequence

from fastapi import HTTPException
from sqlalchemy.orm import Session

from core.inference import embedding_space
from models.document import Document
from models.user import ChatSession

MAX_SELECTED = 20


def owned_session(db: Session, user_id: int, session_id: Optional[int]) -> ChatSession:
    """The chat `session_id` if it belongs to `user_id`, else 404."""
    chat = None
    if session_id is not None:
        chat = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user_id).first()
    if chat is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return chat


def owned_document_ids(db: Session, user_id: int, document_ids: Optional[Sequence[int]],
                       session_id: Optional[int]) -> Optional[list[int]]:
    """The de-duplicated ids if every one belongs to `user_id` AND to chat `session_id`; None when no selection
    was made. A paper of another user or another chat gets the same 404 as a missing one (no existence leak)."""
    if not document_ids:
        return None
    if session_id is None:  # a new chat has no papers (and `== None` would match unassigned legacy papers)
        raise HTTPException(status_code=404, detail="One or more selected documents were not found")
    ids = sorted(set(int(i) for i in document_ids))
    if len(ids) > MAX_SELECTED:
        raise HTTPException(status_code=400, detail=f"select at most {MAX_SELECTED} documents")
    rows = db.query(Document.id, Document.title, Document.filename, Document.status, Document.embedding_model) \
        .filter(Document.user_id == user_id, Document.session_id == session_id, Document.id.in_(ids)).all()
    if {r.id for r in rows} != set(ids):
        raise HTTPException(status_code=404, detail="One or more selected documents were not found")
    space = embedding_space()
    stale = [r.title or r.filename for r in rows if r.status == "completed" and r.embedding_model != space]
    if stale:
        # Never compare a query with vectors from another embedding model (even of the same dimension).
        raise HTTPException(status_code=409, detail=(
            f"{', '.join(stale)}: processed with a different embedding model than the one this server uses "
            f"({space}). Re-process the paper(s) before searching them."))
    return ids
