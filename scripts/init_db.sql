-- ResearchLens schema for a NEW, empty database (the API also creates it on first start).
--
--   psql "$DATABASE_URL" -f scripts/init_db.sql      (optional — not needed for the local quick start)
--
-- DESTRUCTIVE: this drops and recreates every table.
--
-- Matches the SQLAlchemy models in api/models/*.py, including migrations 002_provenance (paper
-- metadata, chunk provenance, cited sources) 003_embedding_space and 004_chat_documents. An EXISTING database must not be
-- reset with this file to pick up those columns — use `python scripts/migrate.py up` instead.

DROP TABLE IF EXISTS chat_messages, chat_sessions, document_chunks, documents, users, documind_migrations CASCADE;

CREATE EXTENSION IF NOT EXISTS vector;

-- Local profile, keyed to a Supabase Auth user by `supabase_uid`.
CREATE TABLE users (
    id           SERIAL PRIMARY KEY,
    supabase_uid TEXT NOT NULL,
    email        TEXT,
    username     TEXT,
    created_at   TIMESTAMPTZ DEFAULT now()
);
CREATE UNIQUE INDEX ix_users_supabase_uid ON users (supabase_uid);
CREATE INDEX        ix_users_email         ON users (email);
CREATE INDEX        ix_users_id            ON users (id);

CREATE TABLE chat_sessions (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER REFERENCES users (id),
    title      TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ix_chat_sessions_id ON chat_sessions (id);

CREATE TABLE chat_messages (
    id         SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES chat_sessions (id),
    role       TEXT,
    content    TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    sources    JSONB              -- assistant turns: numbered sources + citations
);
CREATE INDEX ix_chat_messages_id ON chat_messages (id);

CREATE TABLE documents (
    id       SERIAL PRIMARY KEY,
    user_id  INTEGER REFERENCES users (id),
    filename   TEXT,
    status     TEXT,
    title      TEXT,
    authors    TEXT,
    year       INTEGER,
    page_count INTEGER,
    metadata   JSONB,             -- where each field came from, detected sections
    error      TEXT,              -- short, content-free reason when status = 'failed'
    embedding_model TEXT,         -- embedding space of the vectors, e.g. 'ollama:nomic-embed-text'
    session_id INTEGER            -- the chat the paper was uploaded in (NULL = unassigned legacy paper)
        CONSTRAINT documents_session_id_fkey REFERENCES chat_sessions (id) ON DELETE CASCADE
);
CREATE INDEX ix_documents_id       ON documents (id);
CREATE INDEX ix_documents_filename ON documents (filename);
CREATE INDEX ix_documents_session_id ON documents (session_id);

CREATE TABLE document_chunks (
    id          SERIAL PRIMARY KEY,
    document_id     INTEGER,
    content         TEXT,
    embedding       VECTOR(768),
    page_number     INTEGER,      -- null for plain-text uploads
    section_heading TEXT,
    chunk_index     INTEGER,
    CONSTRAINT document_chunks_document_id_cascade_fkey
        FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
);
CREATE INDEX        ix_document_chunks_id          ON document_chunks (id);
CREATE INDEX        ix_document_chunks_document_id ON document_chunks (document_id);
CREATE UNIQUE INDEX uq_document_chunks_document_chunk_index ON document_chunks (document_id, chunk_index);

CREATE TABLE documind_migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now());
INSERT INTO documind_migrations (name) VALUES ('002_provenance'), ('003_embedding_space'), ('004_chat_documents');

-- Recommended (not created by the app): approximate-NN index for cosine search.
-- Add it once the table has data.
-- CREATE INDEX ix_document_chunks_embedding
--     ON document_chunks USING hnsw (embedding vector_cosine_ops);
