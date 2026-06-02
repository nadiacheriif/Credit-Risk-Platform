"""SQLAlchemy engine, session factory, and schema creation.

PostgreSQL is the system of record everywhere — app and tests. The engine is
built from `settings.database_url` (a postgresql+psycopg2 DSN); the test suite
points this at an ephemeral Postgres container (see tests/conftest.py).
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


# Schema is managed by Alembic migrations (app/db/migrate.py), not create_all().


def get_db():
    """FastAPI dependency yielding a scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
