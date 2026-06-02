"""Run Alembic migrations programmatically (used at app startup and in tests)."""
from __future__ import annotations

from alembic import command
from alembic.config import Config

from app.core.config import ROOT_DIR, settings


def _alembic_config() -> Config:
    cfg = Config(str(ROOT_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", settings.database_url)
    return cfg


def run_migrations() -> None:
    """Upgrade the database to the latest revision (`alembic upgrade head`)."""
    command.upgrade(_alembic_config(), "head")
