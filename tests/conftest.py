"""Pytest fixtures.

PostgreSQL is the system of record, so the tests run against a *real* Postgres —
an ephemeral `postgres:16-alpine` container started via testcontainers and torn
down at the end of the session. The DSN must be exported before any `app.*`
import triggers Settings()/engine creation, so the container is started here at
module import time.

Requires a running Docker daemon. Override with TEST_DATABASE_URL to point the
suite at an already-running Postgres instead (e.g. inside docker-compose).
"""
import os

import pytest

_container = None
_dsn = os.environ.get("TEST_DATABASE_URL")

if not _dsn:
    from testcontainers.postgres import PostgresContainer

    _container = PostgresContainer("postgres:16-alpine")
    _container.start()
    _dsn = _container.get_connection_url()  # postgresql+psycopg2://...

os.environ["DATABASE_URL"] = _dsn
os.environ["MLFLOW_TRACKING_URI"] = ""
os.environ["OLLAMA_BASE_URL"] = ""  # explainer disabled in tests (no LLM dependency)

from fastapi.testclient import TestClient  # noqa: E402

from app.db.migrate import run_migrations  # noqa: E402
from app.main import app  # noqa: E402


def pytest_unconfigure(config):
    if _container is not None:
        _container.stop()


@pytest.fixture(scope="session", autouse=True)
def _setup_db():
    run_migrations()  # exercises the real Alembic migrations against Postgres
    yield


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c
