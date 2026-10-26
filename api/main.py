import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.database import engine, Base
import models.document
import models.user
from routers import document, chat, auth, analysis
from core.observability import configure_logging, request_logging_middleware

configure_logging()

from sqlalchemy import text

with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

Base.metadata.create_all(bind=engine)

# Fail fast on a database older than the models (needs scripts/migrate.py), an unknown
# RETRIEVAL_MODE, or hybrid_bm25 without migration 001_bm25_hybrid.
from core.schema_check import assert_schema_current  # noqa: E402
from core.retrieval import validate_configuration  # noqa: E402

assert_schema_current(engine)
validate_configuration(engine)

# Fail fast on an invalid inference configuration (e.g. INFERENCE_PROVIDER=ollama without OLLAMA_CHAT_MODEL,
# or local embeddings pointed at a hosted database) and report documents from another embedding space.
from core.health import health_report, log_embedding_spaces  # noqa: E402
from core.inference import get_provider  # noqa: E402

get_provider()
log_embedding_spaces(engine)

app = FastAPI(title="ResearchLens API")

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
    # the chat stream reads the new session id and its numbered sources from headers
    expose_headers=["X-Session-ID", "X-Sources", "X-Chat-Mode", "X-Request-ID"],
)
app.middleware("http")(request_logging_middleware)

app.include_router(auth.router, tags=["auth"], prefix="/auth")
app.include_router(document.router, tags=["document"], prefix="/document")
app.include_router(chat.router, tags=["chat"], prefix="/chat")
app.include_router(analysis.router, tags=["analysis"], prefix="/analysis")

@app.get("/")
def read_root():
    return {"message": "ResearchLens API is running"}


@app.get("/health")
def health():
    """Database, broker and inference-server reachability plus the active embedding space (no secrets)."""
    return health_report(engine)
