import os
import sys
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Ensure the api/ directory is importable.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402
from core.database import Base, get_db  # noqa: E402
import core.dependencies as deps  # noqa: E402

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://rag_user:rag_password@localhost:5433/test_rag_engine",
)

engine = create_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client():
    """An unauthenticated TestClient."""
    return TestClient(app)


@pytest.fixture()
def make_auth_client():
    """Factory returning a TestClient authenticated as a given Supabase user.

    Token verification is stubbed (no network), but the real ``get_current_user``
    still runs — so the local-profile upsert is exercised. Each call registers a
    distinct bearer token, so multiple isolated users can coexist in one test.
    """
    registry: dict[str, dict] = {}

    def _verify(token: str) -> dict:
        if token in registry:
            return registry[token]
        raise deps.TokenError("unknown test token")

    with patch.object(deps, "verify_supabase_jwt", side_effect=_verify):

        def _make(sub: str | None = None, email: str = "user@example.com") -> TestClient:
            sub = sub or str(uuid.uuid4())
            token = f"test-{uuid.uuid4().hex}"
            registry[token] = {"sub": sub, "email": email, "exp": 9_999_999_999}
            c = TestClient(app)
            c.headers.update({"Authorization": f"Bearer {token}"})
            return c

        yield _make


@pytest.fixture()
def auth_client(make_auth_client):
    """A TestClient authenticated as one default user."""
    return make_auth_client(email="default@example.com")


@pytest.fixture()
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
