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
    # outside compose. (Tests point this at an ephemeral Postgres container.)
    database_url: str = (
        "postgresql+psycopg2://credit:credit@postgres:5432/credit_risk"
    )

    # MLflow — empty string disables tracking gracefully.
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "credit-risk-inference"

    # Ollama LLM explainer — empty base URL disables it gracefully (the UI then
    # shows only the deterministic reason codes). docker-compose sets this.
    ollama_base_url: str = ""
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_s: float = 30.0

    # Artifact locations
    ml_dir: Path = ROOT_DIR / "ml"
    templates_dir: Path = ROOT_DIR / "templates"
    static_dir: Path = ROOT_DIR / "static"


settings = Settings()
