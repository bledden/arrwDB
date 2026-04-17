CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT,
    content     TEXT NOT NULL,
    embedding   VECTOR(1024),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops);

INSERT INTO documents (title, content, embedding) VALUES
    ('Welcome', 'Welcome to the demo pgvector → arrwDB migration.',
     (SELECT ARRAY(SELECT random()::real FROM generate_series(1, 1024))::vector)),
    ('About', 'arrwDB sits next to Postgres as a vector-search sidecar.',
     (SELECT ARRAY(SELECT random()::real FROM generate_series(1, 1024))::vector));
