"""Hybrid retrieval: RRF fusion (pure) and the SQL path (real Postgres, session-only TEMP tables)."""

import os

import pytest

import hybrid


# --- RRF ----------------------------------------------------------------------------------------
def test_rrf_scores_and_order():
    fused = hybrid.rrf_fuse({"vector": [1, 2, 3], "fts": [3, 4]}, top_k=10, rrf_k=60)
    by_id = {f["chunk_id"]: f for f in fused}
    assert by_id[3]["rrf_score"] == pytest.approx(1 / 63 + 1 / 61)   # in both lists -> highest
    assert [f["chunk_id"] for f in fused] == [3, 1, 2, 4]
    assert by_id[4]["vector_rank"] is None and by_id[4]["fts_rank"] == 2
    assert by_id[1]["fts_rank"] is None and by_id[1]["vector_rank"] == 1


def test_rrf_ties_are_deterministic():
    # 1 and 2 both rank 1 in one list each -> equal score; tie broken by best rank, then id
    fused = hybrid.rrf_fuse({"vector": [2], "fts": [1]}, top_k=2)
    assert [f["chunk_id"] for f in fused] == [1, 2]
    swapped = hybrid.rrf_fuse({"fts": [1], "vector": [2]}, top_k=2)  # list order must not matter
    assert [f["chunk_id"] for f in swapped] == [1, 2]


def test_rrf_top_k_empty_lists_and_validation():
    assert hybrid.rrf_fuse({"vector": [], "fts": []}, top_k=5) == []
    assert len(hybrid.rrf_fuse({"vector": list(range(1, 30))}, top_k=5)) == 5
    with pytest.raises(ValueError):
        hybrid.rrf_fuse({"vector": [1, 1]}, top_k=5)
    with pytest.raises(ValueError):
        hybrid.rrf_fuse({"vector": [1]}, top_k=5, rrf_k=0)


def test_rrf_only_ranks_matter_not_raw_scores():
    a = hybrid.rrf_fuse({"vector": [10, 20, 30], "fts": [30, 20]}, top_k=3)
    b = hybrid.rrf_fuse({"vector": [10, 20, 30], "fts": [30, 20]}, top_k=3, rrf_k=60)
    assert a == b


# --- SQL path on real Postgres: a throwaway database created and dropped by the fixture -----------
@pytest.fixture
def temp_db():
    import uuid
    from urllib.parse import urlparse, urlunparse

    import psycopg2

    import db

    url = os.environ.get("EVAL_DATABASE_URL", db.DEFAULT_EVAL_DATABASE_URL)
    name = f"documind_eval_test_{uuid.uuid4().hex[:8]}"
    parsed = urlparse(url)
    admin_url = urlunparse(parsed._replace(path="/postgres"))
    test_url = urlunparse(parsed._replace(path=f"/{name}"))
    try:
        db.assert_local_eval_database(test_url)
        admin = psycopg2.connect(admin_url, connect_timeout=2)
    except Exception:  # noqa: BLE001 — optional integration test
        pytest.skip("local evaluation database server not reachable")
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute(f"CREATE DATABASE {name}")
    try:
        conn = psycopg2.connect(test_url)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        conn.close()
        conn = db.connect(test_url)  # guarded + pgvector adapter
        with conn.cursor() as cur:
            cur.execute("CREATE TABLE documents (id int PRIMARY KEY, user_id int)")
            cur.execute("CREATE TABLE document_chunks (id int PRIMARY KEY, document_id int, content text, "
                        "embedding vector(3))")
            cur.execute("CREATE TABLE eval_chunk_provenance (chunk_id int, paper_id text, pages int[], "
                        "chunk_ordinal int)")
            cur.execute("INSERT INTO documents VALUES (1, 7), (2, 7), (3, 99)")
            rows = [
                (1, 1, "Diffusion policies model multimodal robot action distributions for the robot.", "[1,0,0]"),
                (2, 1, "We trained on 800k trajectories from Open X-Embodiment.", "[0.9,0.1,0]"),
                (3, 2, "The ALOHA kit costs under 20k USD; CVAE action chunking.", "[0,1,0]"),
                (4, 2, "Unrelated text about kitchens.", "[0.95,0.05,0]"),
                (5, 3, "Other user's chunk about Open X-Embodiment trajectories.", "[1,0,0]"),
                (6, 1, "The robot uses a robot arm and a robot gripper in robot experiments.", "[0.8,0.2,0]"),
            ]
            for cid, did, text, vec in rows:
                cur.execute("INSERT INTO document_chunks VALUES (%s, %s, %s, %s::vector)", (cid, did, text, vec))
                cur.execute("INSERT INTO eval_chunk_provenance VALUES (%s, %s, %s, %s)",
                            (cid, f"paper{did}", [cid], cid))
        conn.commit()
        yield conn
        conn.close()
    finally:
        with admin.cursor() as cur:
            cur.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
        admin.close()


def test_hybrid_sql_scope_fts_and_provenance(temp_db):
    lists = hybrid.candidate_lists(temp_db, user_id=7, query_text="How many trajectories from Open X-Embodiment?",
                                   query_embedding=[1, 0, 0])
    assert 5 not in lists["vector"] and 5 not in lists["fts"]          # other user's chunk never retrieved
    assert lists["vector"][0] == 1 and lists["fts"][0] == 2           # keyword match surfaces chunk 2
    out = hybrid.hybrid_retrieve(temp_db, user_id=7, query_text="How many trajectories from Open X-Embodiment?",
                                 query_embedding=[1, 0, 0], top_k=3)
    assert out[0]["chunk_id"] == 2                                     # in both lists -> fused to the top
    assert out[0]["vector_rank"] is not None and out[0]["fts_rank"] == 1
    assert out[0]["paper_id"] == "paper1" and out[0]["pages"] == [2] and out[0]["rank"] == 1


def test_hybrid_document_filter_and_stopword_query(temp_db):
    out = hybrid.hybrid_retrieve(temp_db, user_id=7, query_text="CVAE action chunking cost",
                                 query_embedding=[1, 0, 0], top_k=5, document_ids=[2])
    assert {r["document_id"] for r in out} == {2}                     # document filter applied
    assert out[0]["chunk_id"] == 3                                     # FTS hit outranks closer vector
    only_stopwords = hybrid.candidate_lists(temp_db, user_id=7, query_text="the of and",
                                            query_embedding=[1, 0, 0])
    assert only_stopwords["fts"] == [] and only_stopwords["vector"]   # degrades to vector-only


# --- BM25 ----------------------------------------------------------------------------------------
def _bm25_reference(conn, query_text: str, chunk_id: int) -> float:
    """Independent Python computation of the BM25 formula from the stored statistics."""
    import math

    with conn.cursor() as cur:
        cur.execute("SELECT n_docs, avgdl FROM eval_bm25_stats")
        n, avgdl = cur.fetchone()
        cur.execute("SELECT DISTINCT unnest(tsvector_to_array(to_tsvector('english', %s)))", (query_text,))
        terms = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT dl FROM eval_bm25_doclen WHERE chunk_id = %s", (chunk_id,))
        dl = cur.fetchone()[0]
        score = 0.0
        for t in terms:
            cur.execute("SELECT count(*) FROM eval_bm25_postings WHERE lexeme = %s", (t,))
            df = cur.fetchone()[0]
            cur.execute("SELECT tf FROM eval_bm25_postings WHERE lexeme = %s AND chunk_id = %s", (t, chunk_id))
            row = cur.fetchone()
            if df and row:
                tf, k1, b = row[0], hybrid.BM25_K1, hybrid.BM25_B
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    return score


def test_bm25_requires_fresh_statistics(temp_db):
    with pytest.raises(hybrid.StaleLexicalIndex, match="missing"):
        hybrid.candidate_lists(temp_db, user_id=7, query_text="robot", query_embedding=[1, 0, 0], lexical="bm25")
    info = hybrid.build_lexical_index(temp_db)
    assert info["n_docs"] == 6 and info["postings"] > 0
    hybrid.candidate_lists(temp_db, user_id=7, query_text="robot", query_embedding=[1, 0, 0], lexical="bm25")
    with temp_db.cursor() as cur:  # corpus changes -> statistics are stale until rebuilt
        cur.execute("INSERT INTO document_chunks VALUES (7, 1, 'new chunk about grippers', '[1,0,0]'::vector)")
    temp_db.commit()
    with pytest.raises(hybrid.StaleLexicalIndex, match="stale"):
        hybrid.candidate_lists(temp_db, user_id=7, query_text="robot", query_embedding=[1, 0, 0], lexical="bm25")
    assert hybrid.build_lexical_index(temp_db)["n_docs"] == 7


def test_bm25_statistics_and_scores_match_the_formula(temp_db):
    hybrid.build_lexical_index(temp_db)
    with temp_db.cursor() as cur:
        cur.execute("SELECT tf FROM eval_bm25_postings WHERE chunk_id = 6 AND lexeme = 'robot'")
        assert cur.fetchone()[0] == 4                                  # term frequency from tsvector positions
        cur.execute("SELECT count(*) FROM eval_bm25_postings WHERE lexeme = 'robot'")
        assert cur.fetchone()[0] == 2                                  # document frequency (chunks 1 and 6)
    query = "robot trajectories Open X-Embodiment"
    with temp_db.cursor() as cur:
        params = {"user_id": 7, "query_text": query, "candidates": 10, "document_ids": None,
                  "query_embedding": [1, 0, 0], "k1": hybrid.BM25_K1, "b": hybrid.BM25_B}
        cur.execute(hybrid.BM25_SQL, params)
        scored = {cid: float(score) for cid, score in cur.fetchall()}
    for cid, score in scored.items():
        assert score == pytest.approx(_bm25_reference(temp_db, query, cid), rel=1e-9)
    assert 5 not in scored                                             # user scope applies to BM25 too


def test_bm25_rare_terms_beat_frequent_terms(temp_db):
    """The F-14 failure mode: with ts_rank_cd a chunk repeating a common word can win; BM25's IDF
    prefers the chunk matching the rare, specific terms."""
    hybrid.build_lexical_index(temp_db)
    query = "robot trajectories Open X-Embodiment"
    bm25 = hybrid.candidate_lists(temp_db, user_id=7, query_text=query, query_embedding=[0, 0, 1], lexical="bm25")
    assert bm25["fts"][0] == 2                                         # specific evidence first
    out = hybrid.hybrid_retrieve(temp_db, user_id=7, query_text=query, query_embedding=[0, 0, 1], top_k=3,
                                 lexical="bm25")
    assert out[0]["chunk_id"] == 2 and out[0]["fts_rank"] == 1 and out[0]["paper_id"] == "paper1"
