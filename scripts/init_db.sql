-- DocuMind schema — run once against the Supabase database.
--
--   Supabase dashboard -> SQL Editor -> paste -> Run
--   or:  psql "$DATABASE_URL" -f scripts/init_db.sql
--
-- DESTRUCTIVE: this drops and recreates every table. The move to Supabase Auth
-- changed the `users` table shape and existing accounts have no Supabase
-- identity, so a reset is expected (see the plan / README).
--
-- Matches the SQLAlchemy models in api/models/*.py.

DROP TABLE IF EXISTS chat_messages, chat_sessions, document_chunks, documents, users CASCADE;

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
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ix_chat_messages_id ON chat_messages (id);

CREATE TABLE documents (
    id       SERIAL PRIMARY KEY,
    user_id  INTEGER REFERENCES users (id),
    filename TEXT,
    status   TEXT
);
CREATE INDEX ix_documents_id       ON documents (id);
CREATE INDEX ix_documents_filename ON documents (filename);

CREATE TABLE document_chunks (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents (id),
    content     TEXT,
    embedding   VECTOR(768)
);
CREATE INDEX ix_document_chunks_id ON document_chunks (id);

-- Recommended (not created by the app): approximate-NN index for cosine search.
-- Add it once the table has data.
-- CREATE INDEX ix_document_chunks_embedding
--     ON document_chunks USING hnsw (embedding vector_cosine_ops);
