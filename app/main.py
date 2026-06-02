"""FastAPI application entrypoint — serves both the JSON API and the web UI."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.core.config import settings
from app.db.migrate import run_migrations
from app.ui.routes import router as ui_router
from ml.inference import get_engine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("credit-risk")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Apply DB migrations and warm the model artifacts at startup.
    run_migrations()
    log.info("Database migrated to head")
    try:
        get_engine()
        log.info("Model artifacts loaded (%s)", settings.model_version)
    except Exception as exc:  # pragma: no cover
        log.error("Failed to load model artifacts: %s", exc)
    yield


app = FastAPI(title=settings.app_name, version=settings.model_version, lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
app.include_router(api_router)   # /api/*
app.include_router(ui_router)    # /, /predict, /applications, /health
