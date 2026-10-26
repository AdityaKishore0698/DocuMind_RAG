import logging
import time
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from core.database import get_db
from models.document import Document
from models.user import User, ChatSession, ChatMessage
from core.dependencies import get_current_user
from core.inference import embedding_space, get_provider
from core.retrieval import default_top_k, retrieval_mode, retrieve_chunks
from core.ownership import MAX_SELECTED, owned_document_ids, owned_session
from core.storage import object_key, remove_stored_files
from core.sources import PROMPT_RULES, build_sources, encode_header, parse_citations, source_label, user_turn
from core import agent as agent_core

router = APIRouter()
log = logging.getLogger("documind.chat")

NO_EVIDENCE = ("I couldn't find anything in your uploaded papers that addresses this question, so I can't answer "
               "it from your documents. Try selecting different papers or rephrasing.")
NO_PAPERS = ("This chat has no papers yet. Papers belong to the chat they are uploaded in — upload PDFs in this chat, "
             "then ask again.")
NEW_CHAT_TITLE = "New chat"
OTHER_SPACE = ("None of your papers can be searched with the current embedding model ({space}): {n} paper(s) were "
               "processed with a different one. Use “Re-process” in the sidebar, then ask again.")
GENERATION_FAILED = "\n\n⚠️ The answer could not be completed because the language model is unavailable. Please try again."


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    session_id: Optional[int] = None
    # Restrict retrieval to these papers (ownership-checked); omitted = all of the user's papers.
    document_ids: Optional[List[int]] = Field(default=None, max_length=MAX_SELECTED)
    # "agent" = bounded LangGraph evidence gathering before the answer (AGENT_ENABLED only).
    mode: Literal["rag", "agent"] = "rag"

class SessionCreate(BaseModel):
    title: str

def get_embedding(text: str) -> List[float]:
    # Gemini by default; INFERENCE_PROVIDER=ollama uses nomic-embed-text (see core/inference.py).
    return get_provider().embed(text, kind="query")


def _unavailable(what: str, exc: Exception) -> HTTPException:
    quota = any(s in str(exc) for s in ("429", "RESOURCE_EXHAUSTED", "quota"))
    log.warning("event=inference_error step=%s error=%s quota=%s", what, type(exc).__name__, quota)
    detail = ("The AI service quota is exhausted for now. Please try again later." if quota else
              "The AI service is temporarily unavailable. Please try again shortly.")
    return HTTPException(status_code=503, detail=detail)


@router.get("/capabilities")
def capabilities(current_user: User = Depends(get_current_user)):
    return {"agent": agent_core.agent_enabled(), "retrieval_mode": retrieval_mode(), "top_k": default_top_k(),
            "max_selected": MAX_SELECTED, "embedding_space": embedding_space()}

@router.post("/sessions")
def create_session(session: SessionCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_session = ChatSession(user_id=current_user.id, title=session.title)
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return {"id": db_session.id, "title": db_session.title}

@router.get("/sessions")
def list_sessions(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    sessions = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).order_by(ChatSession.created_at.desc()).all()
    return [{"id": s.id, "title": s.title, "created_at": s.created_at, "paper_count": len(s.documents)}
            for s in sessions]

@router.put("/sessions/{session_id}")
def rename_session(session_id: int, session: SessionCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    db_session.title = session.title
    db.commit()
    return {"id": db_session.id, "title": db_session.title}

@router.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    return [{"id": m.id, "role": m.role, "content": m.content, "sources": m.sources} for m in db_session.messages]

@router.delete("/sessions/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_session = owned_session(db, current_user.id, session_id)
    # The chat's papers are deleted with it (ORM cascade; the FK also cascades): only papers of this user in this chat.
    papers = db.query(Document).filter(Document.session_id == db_session.id, Document.user_id == current_user.id).all()
    keys = [object_key(d.user_id, d.id, d.filename) for d in papers]
    db.delete(db_session)
    db.commit()
    # Stored files go after the rows are committed; failures are reported, not swallowed.
    not_removed = remove_stored_files(keys)
    return {"message": "Session deleted", "papers_deleted": len(keys), "files_not_removed": not_removed}


def _agent_context(db: Session, user_id: int, session_id: int, query: str, selected: Optional[list[int]]):
    """Runs the bounded LangGraph workflow (in a worker thread) over the chat's papers; returns (chunks, trace)."""
    provider = get_provider()
    top_k = max(default_top_k(), agent_core.PER_QUERY_K)

    def retrieve(q: str, doc_ids):
        emb = query_embedding if q == query else get_embedding(q)
        return retrieve_chunks(db, user_id=user_id, session_id=session_id, query_text=q, query_embedding=emb,
                               top_k=top_k, document_ids=doc_ids)

    query_embedding = get_embedding(query)
    deps = agent_core.AgentDeps(
        retrieve=retrieve, llm_json=provider.complete_json,
        label=lambda c: source_label({"title": c.document.title or c.document.filename if c.document else None,
                                      "document_id": c.document_id, "page_number": c.page_number,
                                      "section": c.section_heading}))
    result = agent_core.run_agent(deps, query, selected)
    trace = {"rounds": result.rounds, "queries": len(result.queries), "llm_calls": result.llm_calls,
             "missing_documents": result.missing_documents, "elapsed_s": round(result.elapsed_s, 2)}
    return result.context, trace


@router.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    started = time.perf_counter()
    if request.mode == "agent" and not agent_core.agent_enabled():
        raise HTTPException(status_code=400, detail="Deep analysis mode is not enabled on this server")
    # Papers are chat-scoped: the chat must be the user's, and every selected paper must belong to that chat.
    db_session = owned_session(db, current_user.id, request.session_id) if request.session_id else None
    scope = db_session.id if db_session else None
    selected = owned_document_ids(db, current_user.id, request.document_ids, scope)
    has_papers = scope is not None and db.query(Document.id).filter(
        Document.user_id == current_user.id, Document.session_id == scope).first() is not None

    # Evidence first: an unavailable embedding service fails the request cleanly, before anything is stored.
    mode, trace, results = request.mode, None, []
    if not has_papers:
        mode = "rag"  # nothing to retrieve: answer with NO_PAPERS without calling any model
    elif mode == "agent":
        try:
            results, trace = await run_in_threadpool(_agent_context, db, current_user.id, scope, request.query,
                                                     selected)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — the graph failing degrades to plain RAG
            log.warning("event=agent_fallback error=%s", type(exc).__name__)
            db.rollback()
            mode = "rag"
    if mode == "rag" and has_papers:
        try:
            query_embedding = get_embedding(request.query)
        except Exception as exc:  # noqa: BLE001
            raise _unavailable("query_embedding", exc) from exc
        # Only chunks of this user's papers in this chat (optionally the selected ones);
        # vector-only unless RETRIEVAL_MODE=hybrid_bm25.
        results = retrieve_chunks(db, user_id=current_user.id, session_id=scope, query_text=request.query,
                                  query_embedding=query_embedding, document_ids=selected)

    if db_session is None:
        db_session = ChatSession(user_id=current_user.id, title=request.query[:30])
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
    elif db_session.title == NEW_CHAT_TITLE and not db_session.messages:
        db_session.title = request.query[:30]  # a chat created by its first upload is named by its first question
        db.commit()
    session_id = db_session.id

    # Prior turns (before saving the current query).
    history = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at).all()
    past_turns = [(msg.role, msg.content) for msg in history]
    db.add(ChatMessage(session_id=session_id, role="user", content=request.query))
    db.commit()

    sources = build_sources(results)
    system_prompt = ("You are ResearchLens, an assistant that answers questions about the user's uploaded research "
                     "papers. " + PROMPT_RULES)
    question = user_turn(sources, results, request.query)
    provider = get_provider()
    no_evidence = NO_EVIDENCE if has_papers else NO_PAPERS
    if not results and has_papers:
        space = embedding_space()
        mine = db.query(Document).filter(Document.user_id == current_user.id, Document.session_id == session_id,
                                         Document.status == "completed")
        stale = mine.filter(Document.embedding_model.is_distinct_from(space)).count()
        searchable = mine.filter(Document.embedding_model == space).count()
        if stale and not searchable:
            no_evidence = OTHER_SPACE.format(space=space, n=stale)

    async def generate_response():
        full_response, failed = "", False
        try:
            if not results:
                full_response = no_evidence
                yield no_evidence
                return
            try:
                async for text in provider.stream_chat(system_prompt, past_turns, question):
                    full_response += text
                    yield text
            except Exception as exc:  # noqa: BLE001 — end the stream with a notice, not a broken connection
                failed = True
                log.warning("event=generation_error error=%s", type(exc).__name__)
                full_response += GENERATION_FAILED
                yield GENERATION_FAILED
        finally:
            if full_response:
                # Save assistant message to DB even if aborted early
                citations = parse_citations(full_response, len(sources))
                record = {"sources": sources, "cited": citations["cited"], "invalid": citations["invalid"],
                          "mode": mode, "failed": failed}
                if trace:
                    record["agent"] = trace
                db.add(ChatMessage(session_id=session_id, role="assistant", content=full_response, sources=record))
                db.commit()
                log.info("event=chat_done mode=%s model=%s sources=%s cited=%s invalid=%s failed=%s latency_ms=%.0f",
                         mode, getattr(provider, "chat_model", "?"), len(sources), len(citations["cited"]), len(citations["invalid"]),
                         failed, (time.perf_counter() - started) * 1000)

    # Pass the session_id in headers so the frontend knows what session was created
    headers = {"X-Session-ID": str(session_id), "X-Sources": encode_header(sources), "X-Chat-Mode": mode}
    return StreamingResponse(generate_response(), media_type="text/plain", headers=headers)
