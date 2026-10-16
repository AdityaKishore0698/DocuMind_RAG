import pytest

from db import UnsafeDatabaseError, assert_local_eval_database, redacted

OK = "postgresql://postgres:postgres@127.0.0.1:5434/documind_eval"


def test_accepts_local_eval_database():
    assert_local_eval_database(OK)
    assert_local_eval_database("postgresql://u:p@localhost:5432/my_eval_db")


@pytest.mark.parametrize("url", [
    "postgresql://postgres.abc:pw@aws-0-eu.pooler.supabase.com:6543/postgres",   # production-style pooler
    "postgresql://postgres:pw@db.abcdefgh.supabase.co:5432/documind_eval",      # supabase host
    "postgresql://u:p@10.0.0.5:5432/documind_eval",                              # remote host
    "postgresql://u:p@127.0.0.1:5432/postgres",                                  # not an eval db name
    "mysql://u:p@127.0.0.1/documind_eval",                                       # wrong scheme
])
def test_rejects_anything_that_could_be_production(url):
    with pytest.raises(UnsafeDatabaseError):
        assert_local_eval_database(url)


def test_rejects_url_equal_to_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", OK)
    with pytest.raises(UnsafeDatabaseError):
        assert_local_eval_database(OK)


def test_redacted_hides_credentials():
    assert "postgres:postgres" not in redacted(OK)
    assert redacted(OK).endswith("127.0.0.1:5434/documind_eval")


def test_vector_to_list_accepts_every_adapter_return_type():
    import numpy as np
    from pgvector import Vector

    from db import vector_to_list

    expected = [1.0, 2.5, -3.0]
    assert vector_to_list(Vector(expected)) == expected          # pgvector object (not iterable)
    assert vector_to_list(np.asarray(expected, dtype=np.float32)) == expected
    assert vector_to_list(expected) == expected


def test_cached_embeddings_round_trip_on_real_db_when_available():
    import os

    import psycopg2

    from db import DEFAULT_EVAL_DATABASE_URL, cached_embeddings, connect, store_embeddings

    url = os.environ.get("EVAL_DATABASE_URL", DEFAULT_EVAL_DATABASE_URL)
    try:
        conn = connect(url)
    except (psycopg2.Error, Exception):  # noqa: BLE001 — optional integration check
        import pytest

        pytest.skip("local evaluation database not reachable")
    try:
        vec = [0.25] * 768
        store_embeddings(conn, "unit-test-model", 768, [("cache round trip probe", vec)])
        got = cached_embeddings(conn, "unit-test-model", 768, ["cache round trip probe"])
        assert list(got.values())[0] == vec
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM eval_embedding_cache WHERE model = 'unit-test-model'")
        conn.commit()
        conn.close()
