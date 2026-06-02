"""Application configuration (env-driven)."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (app/core/config.py -> project root)
ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Credit Risk Platform"
    model_version: str = "campari-woe-lr-1.0.0"

    # Database — PostgreSQL is the system of record. This default matches the
    # docker-compose `postgres` service; override with DATABASE_URL when running
    # outside compose. (The fast unit tests point this at sqlite via env.)
    database_url: str = (
        "postgresql+psycopg2://credit:credit@postgres:5432/credit_risk"
    )

    # MLflow — empty string disables tracking gracefully.
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "credit-risk-inference"

    # Artifact locations
    ml_dir: Path = ROOT_DIR / "ml"
    templates_dir: Path = ROOT_DIR / "templates"
    static_dir: Path = ROOT_DIR / "static"


settings = Settings()
