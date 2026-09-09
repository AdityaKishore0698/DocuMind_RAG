import os
from typing import Optional, List

from google import genai
from google.genai import types
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from core.database import get_db
from models.document import DocumentChunk, Document
from models.user import User, ChatSession, ChatMessage
from core.dependencies import get_current_user

router = APIRouter()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.6-flash")
EMBEDDING_DIM = 768

# Instantiated lazily so the app (and test suite) can import without GEMINI_API_KEY set.
_genai_client: Optional[genai.Client] = None


def get_genai_client() -> genai.Client:
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _genai_client


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[int] = None

class SessionCreate(BaseModel):
    title: str

def get_embedding(text: str) -> List[float]:
    result = get_genai_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    return result.embeddings[0].values

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
    return [{"id": s.id, "title": s.title, "created_at": s.created_at} for s in sessions]

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
    return [{"role": m.role, "content": m.content} for m in db_session.messages]

@router.delete("/sessions/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    db_session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(db_session)
    db.commit()
    return {"message": "Session deleted"}

@router.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    session_id = request.session_id
    if not session_id:
        db_session = ChatSession(user_id=current_user.id, title=request.query[:30])
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        session_id = db_session.id
    else:
        db_session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
        if not db_session:
            raise HTTPException(status_code=404, detail="Session not found")

    # Save user message
    user_message = ChatMessage(session_id=session_id, role="user", content=request.query)
    db.add(user_message)
    db.commit()

    query_embedding = get_embedding(request.query)
    
    # RAG Vector Search: Only search chunks from documents belonging to the current user!
    results = db.query(DocumentChunk).join(Document).filter(
        Document.user_id == current_user.id
    ).order_by(
        DocumentChunk.embedding.cosine_distance(query_embedding)
    ).limit(5).all()
    
    context = "\n\n".join([chunk.content for chunk in results])
    
    # Load past history
    history = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at).all()

    system_prompt = (
        "You are DocuMind, an assistant that answers questions using the user's "
        "uploaded documents. Rely on the context below; if it does not contain the "
        "answer, say so.\n\n"
        f"Context from documents:\n{context}"
    )

    # Gemini expects "user" / "model" roles; our history stores "user" / "assistant".
    contents: List[types.Content] = [
        types.Content(
            role="model" if msg.role == "assistant" else "user",
            parts=[types.Part(text=msg.content)],
        )
        for msg in history[:-1]  # exclude the current query we just added
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=request.query)]))

    async def generate_response():
        full_response = ""
        try:
            stream = await get_genai_client().aio.models.generate_content_stream(
                model=CHAT_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
            )
            async for chunk in stream:
                if chunk.text:
                    full_response += chunk.text
                    yield chunk.text
        finally:
            if full_response:
                # Save assistant message to DB even if aborted early
                assistant_message = ChatMessage(session_id=session_id, role="assistant", content=full_response)
                db.add(assistant_message)
                db.commit()

    # Pass the session_id in headers so the frontend knows what session was created
    return StreamingResponse(generate_response(), media_type="text/plain", headers={"X-Session-ID": str(session_id)})
