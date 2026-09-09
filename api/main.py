import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.database import engine, Base
import models.document
import models.user
from routers import document, chat, auth

from sqlalchemy import text

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

Base.metadata.create_all(bind=engine)

app = FastAPI(title="DocuMind API")

# Allow the Next.js frontend (browser) to call the API cross-origin.
# FRONTEND_ORIGINS is a comma-separated list; defaults to the local dev servers
# (Next.js picks :3001 automatically when :3000 is taken). Set it to "*" to allow
# any origin during local development.
frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://localhost:3001",
    ).split(",")
    if origin.strip()
]
allow_all_origins = "*" in frontend_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else frontend_origins,
    # A wildcard origin cannot be combined with credentials per the CORS spec;
    # this API authenticates via the Authorization header, not cookies, so that's fine.
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-ID"],  # so the chat stream can read the new session id
)

app.include_router(auth.router, tags=["auth"], prefix="/auth")
app.include_router(document.router, tags=["document"], prefix="/document")
app.include_router(chat.router, tags=["chat"], prefix="/chat")

@app.get("/")
def read_root():
    return {"message": "DocuMind API is running"}
