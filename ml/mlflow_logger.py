"""
Lightweight MLflow logging for inference requests.

Tracking is best-effort: if MLFLOW_TRACKING_URI is unset or the server is
unreachable, prediction must still succeed. All failures are swallowed.
"""
from __future__ import annotations

import logging

from app.core.config import settings

log = logging.getLogger("mlflow_logger")

_enabled = bool(settings.mlflow_tracking_uri)
_mlflow = None

if _enabled:
    try:
        import mlflow as _mlflow  # noqa: N813

        _mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        _mlflow.set_experiment(settings.mlflow_experiment)
        log.info("MLflow tracking enabled at %s", settings.mlflow_tracking_uri)
    except Exception as exc:  # pragma: no cover - infra dependent
        log.warning("MLflow disabled (init failed): %s", exc)
        _enabled = False


def log_inference(application: dict, result: dict, latency_ms: float) -> None:
    """Log one inference as an MLflow run. Never raises."""
    if not _enabled or _mlflow is None:
        return
    try:
        with _mlflow.start_run(run_name="inference"):
            _mlflow.set_tag("model_version", result["model_version"])
            _mlflow.set_tag("decision", result["decision"])
            _mlflow.set_tag("risk_grade", result["risk_grade"])
            _mlflow.log_param("loan_purpose", application.get("purpose"))
            _mlflow.log_param("credit_amount", application.get("credit_amount"))
            _mlflow.log_param("duration", application.get("duration"))
            _mlflow.log_metric("probability_default", result["probability_default"])
            _mlflow.log_metric("probability_good", result["probability_good"])
            _mlflow.log_metric("credit_score", result["credit_score"])
            _mlflow.log_metric("latency_ms", latency_ms)
    except Exception as exc:  # pragma: no cover - infra dependent
        log.warning("MLflow logging skipped: %s", exc)
