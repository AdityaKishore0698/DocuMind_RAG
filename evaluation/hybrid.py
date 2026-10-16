"""Hybrid retrieval (Phase 2): pgvector cosine search + PostgreSQL full-text search, fused with
Reciprocal Rank Fusion (RRF).

Both candidate lists are scoped exactly like System A (``documents.user_id``) plus an optional
``document_ids`` filter (an additional restriction, never a replacement for the user scope).

* Vector list   — ``embedding <=> query`` ascending (same operator as System A), top ``candidates``.
* Full-text list — the question's English lexemes OR-ed into one tsquery (``'a' | 'b' | …``), ranked by
  ``ts_rank_cd`` over ``to_tsvector('english', content)``. OR (not AND) because natural-language
  questions rarely have every term in one chunk. Lexemes are cast straight to ``tsquery`` so hyphenated
  terms (``x-embodi``) stay single lexemes. Computed at query time — no stored tsvector or GIN index in
  the evaluation DB; production would need a migration for both (see docs/technical-decisions.md).
* Fusion — ``score(d) = Σ_lists 1 / (rrf_k + rank_list(d))`` with ``rrf_k`` = 60 (Cormack et al., 2009).
  Ties break deterministically on best single-list rank, then chunk id. Raw scores (cosine distance,
  ts_rank) are never mixed — only ranks — which is why RRF needs no score normalisation.

Every result carries its rank in each list (``vector_rank`` / ``fts_rank``, None if absent) and the
fused ``rrf_score``, so ranking decisions are inspectable.

BM25 variant (``lexical="bm25"``) — the same FTS match (``@@``, GIN expression index) but ranked by Okapi
BM25 instead of ``ts_rank_cd``, which has no IDF and lets corpus-frequent terms (``robot`` in 66 % of
chunks) dominate OR-queries (docs/failure-analysis.md F-14):

    score(d, q) = Σ_{t ∈ q} IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
    IDF(t)      = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))            (Lucene's non-negative form)

k1 = 1.2 and b = 0.75 are the conventional defaults, fixed a priori (not tuned on the benchmark).
tf, |d| (token count) and df come from the same ``to_tsvector('english', content)`` the match uses,
materialised by ``build_lexical_index`` into harness tables; a corpus fingerprint makes stale statistics
an error instead of a silent mis-ranking.
"""

from __future__ import annotations

RRF_K = 60
CANDIDATES = 20
BM25_K1 = 1.2
BM25_B = 0.75

_SCOPE = ("documents.user_id = %(user_id)s "
          "AND (%(document_ids)s::int[] IS NULL OR document_chunks.document_id = ANY(%(document_ids)s::int[]))")

VECTOR_SQL = f"""
SELECT document_chunks.id
FROM document_chunks
JOIN documents ON documents.id = document_chunks.document_id
WHERE {_SCOPE}
ORDER BY document_chunks.embedding <=> %(query_embedding)s, document_chunks.id
LIMIT %(candidates)s
"""

FTS_SQL = f"""
WITH q AS (
    SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS tsq
    FROM unnest(tsvector_to_array(to_tsvector('english', %(query_text)s))) AS lexeme
)
SELECT document_chunks.id
FROM document_chunks
JOIN documents ON documents.id = document_chunks.document_id
CROSS JOIN q
WHERE {_SCOPE}
  AND q.tsq IS NOT NULL
  AND to_tsvector('english', document_chunks.content) @@ q.tsq
ORDER BY ts_rank_cd(to_tsvector('english', document_chunks.content), q.tsq) DESC, document_chunks.id
LIMIT %(candidates)s
"""


BM25_SQL = f"""
WITH qlex AS (
    SELECT DISTINCT lexeme FROM unnest(tsvector_to_array(to_tsvector('english', %(query_text)s))) AS lexeme
), q AS (
    SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS tsq FROM qlex
), matched AS (
    SELECT document_chunks.id
    FROM document_chunks
    JOIN documents ON documents.id = document_chunks.document_id
    CROSS JOIN q
    WHERE {_SCOPE}
      AND q.tsq IS NOT NULL
      AND to_tsvector('english', document_chunks.content) @@ q.tsq
), df AS (
    SELECT p.lexeme, count(*) AS df FROM eval_bm25_postings p JOIN qlex USING (lexeme) GROUP BY p.lexeme
)
SELECT m.id,
       sum(ln(1 + (s.n_docs - df.df + 0.5) / (df.df + 0.5))
           * p.tf * (%(k1)s + 1) / (p.tf + %(k1)s * (1 - %(b)s + %(b)s * dl.dl / s.avgdl))) AS score
FROM matched m
JOIN eval_bm25_postings p ON p.chunk_id = m.id
JOIN qlex USING (lexeme)
JOIN df USING (lexeme)
JOIN eval_bm25_doclen dl ON dl.chunk_id = m.id
CROSS JOIN eval_bm25_stats s
GROUP BY m.id
ORDER BY score DESC, m.id
LIMIT %(candidates)s
"""

LEXICAL_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS eval_document_chunks_fts_gin ON document_chunks USING gin (to_tsvector('english', content));
CREATE TABLE IF NOT EXISTS eval_bm25_postings (lexeme TEXT NOT NULL, chunk_id INTEGER NOT NULL, tf INTEGER NOT NULL,
                                               PRIMARY KEY (lexeme, chunk_id));
CREATE TABLE IF NOT EXISTS eval_bm25_doclen (chunk_id INTEGER PRIMARY KEY, dl INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS eval_bm25_stats (n_docs INTEGER NOT NULL, avgdl DOUBLE PRECISION NOT NULL,
                                            corpus_fingerprint TEXT NOT NULL, built_at TIMESTAMPTZ NOT NULL);
"""

FINGERPRINT_SQL = ("SELECT count(*) || ':' || coalesce(md5(string_agg(id::text || ':' || md5(content), ',' ORDER BY id)), '') "
                   "FROM document_chunks")


class StaleLexicalIndex(RuntimeError):
    """BM25 statistics do not describe the current corpus — rebuild them."""


def build_lexical_index(conn) -> dict:
    """(Re)build the GIN index and BM25 statistics in ONE transaction from the current corpus.
    tf = number of positions of a lexeme in the chunk's tsvector; |d| = sum of tf (tokens after stop-word
    removal); df = chunks containing the lexeme. Idempotent; call after every ingestion change."""
    with conn.cursor() as cur:
        cur.execute(LEXICAL_INDEX_DDL)
        cur.execute("TRUNCATE eval_bm25_postings, eval_bm25_doclen, eval_bm25_stats")
        cur.execute("""
            INSERT INTO eval_bm25_postings (lexeme, chunk_id, tf)
            SELECT (t).lexeme, c.id, coalesce(array_length((t).positions, 1), 1)
            FROM document_chunks c CROSS JOIN LATERAL unnest(to_tsvector('english', c.content)) AS t""")
        cur.execute("""
            INSERT INTO eval_bm25_doclen (chunk_id, dl)
            SELECT c.id, coalesce((SELECT sum(tf) FROM eval_bm25_postings p WHERE p.chunk_id = c.id), 0)
            FROM document_chunks c""")
        cur.execute(FINGERPRINT_SQL)
        fingerprint = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO eval_bm25_stats (n_docs, avgdl, corpus_fingerprint, built_at)
            SELECT count(*), coalesce(avg(dl), 0), %s, now() FROM eval_bm25_doclen""", (fingerprint,))
        cur.execute("ANALYZE eval_bm25_postings; ANALYZE eval_bm25_doclen; ANALYZE document_chunks")
        cur.execute("SELECT n_docs, avgdl FROM eval_bm25_stats")
        n_docs, avgdl = cur.fetchone()
        cur.execute("SELECT count(*), count(DISTINCT lexeme) FROM eval_bm25_postings")
        postings, vocabulary = cur.fetchone()
    conn.commit()
    return {"n_docs": n_docs, "avgdl": round(avgdl, 2), "postings": postings, "vocabulary": vocabulary,
            "fingerprint": fingerprint}


def assert_lexical_index_fresh(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('eval_bm25_stats')")
        if cur.fetchone()[0] is None:
            raise StaleLexicalIndex("BM25 statistics missing — run `run_eval.py build-lexical-index`")
        cur.execute("SELECT corpus_fingerprint FROM eval_bm25_stats")
        row = cur.fetchone()
        cur.execute(FINGERPRINT_SQL)
        current = cur.fetchone()[0]
    if row is None or row[0] != current:
        raise StaleLexicalIndex("BM25 statistics are stale (corpus changed) — run `run_eval.py build-lexical-index`")


def rrf_fuse(ranked_lists: dict[str, list[int]], top_k: int, rrf_k: int = RRF_K) -> list[dict]:
    """Fuse ranked id lists. Returns [{"chunk_id", "rrf_score", "<name>_rank", ...}] best first."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    ranks: dict[int, dict[str, int]] = {}
    for name, ids in ranked_lists.items():
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate ids in ranked list {name!r}")
        for rank, chunk_id in enumerate(ids, start=1):
            ranks.setdefault(chunk_id, {})[name] = rank
    fused = []
    for chunk_id, by_list in ranks.items():
        score = sum(1.0 / (rrf_k + r) for r in by_list.values())
        fused.append({"chunk_id": chunk_id, "rrf_score": score,
                      **{f"{name}_rank": by_list.get(name) for name in ranked_lists}})
    fused.sort(key=lambda d: (-d["rrf_score"], min(r for r in (d.get(f"{n}_rank") for n in ranked_lists) if r),
                              d["chunk_id"]))
    return fused[:top_k]


def candidate_lists(conn, *, user_id: int, query_text: str, query_embedding: list[float],
                    document_ids: list[int] | None = None, candidates: int = CANDIDATES,
                    lexical: str = "tsrank") -> dict[str, list[int]]:
    """Vector and lexical candidate ids. ``lexical`` = "tsrank" (ts_rank_cd) or "bm25"."""
    import numpy as np

    if lexical not in ("tsrank", "bm25"):
        raise ValueError("lexical must be 'tsrank' or 'bm25'")
    if lexical == "bm25":
        assert_lexical_index_fresh(conn)
    params = {"user_id": user_id, "query_text": query_text, "candidates": candidates,
              "document_ids": list(document_ids) if document_ids else None,
              "query_embedding": np.asarray(query_embedding, dtype=np.float32), "k1": BM25_K1, "b": BM25_B}
    with conn.cursor() as cur:
        cur.execute(VECTOR_SQL, params)
        vector_ids = [r[0] for r in cur.fetchall()]
        cur.execute(BM25_SQL if lexical == "bm25" else FTS_SQL, params)
        fts_ids = [r[0] for r in cur.fetchall()]
    return {"vector": vector_ids, "fts": fts_ids}


def hybrid_retrieve(conn, *, user_id: int, query_text: str, query_embedding: list[float], top_k: int,
                    document_ids: list[int] | None = None, candidates: int = CANDIDATES,
                    rrf_k: int = RRF_K, lexical: str = "tsrank") -> list[dict]:
    """Top-k fused chunks in the same shape as db.retrieve (plus per-list ranks and rrf_score)."""
    lists = candidate_lists(conn, user_id=user_id, query_text=query_text, query_embedding=query_embedding,
                            document_ids=document_ids, candidates=candidates, lexical=lexical)
    fused = rrf_fuse(lists, top_k=top_k, rrf_k=rrf_k)
    ids = [f["chunk_id"] for f in fused]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.id, c.document_id, c.content, p.paper_id, p.pages, p.chunk_ordinal "
            "FROM document_chunks c LEFT JOIN eval_chunk_provenance p ON p.chunk_id = c.id WHERE c.id = ANY(%s)",
            (ids,))
        rows = {r[0]: r for r in cur.fetchall()}
    results = []
    for rank, f in enumerate(fused, start=1):
        _, document_id, content, paper_id, pages, ordinal = rows[f["chunk_id"]]
        results.append({"rank": rank, "chunk_id": f["chunk_id"], "document_id": document_id, "paper_id": paper_id,
                        "pages": list(pages or []), "chunk_ordinal": ordinal, "text": content,
                        "rrf_score": f["rrf_score"], "vector_rank": f.get("vector_rank"),
                        "fts_rank": f.get("fts_rank")})
    return results
