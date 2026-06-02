"""
LLM explanation layer (local Ollama) — turns the deterministic reason codes into
a plain-English narrative.

Hard rules:
  * The LLM NEVER makes or changes the decision. It only rephrases the reason
    codes produced by the scorecard.
  * Best-effort: if Ollama is unset/unreachable/slow, return None and the UI
    falls back to the deterministic reason codes. It must never raise.
  * Grounded: the model receives only the structured reasons + decision (feature
    labels, not raw PII beyond what the reasons already contain) and is told it
    may not invent or omit reasons.
"""
from __future__ import annotations

import logging

import httpx

from app.core.config import settings

log = logging.getLogger("explainer")

enabled = bool(settings.ollama_base_url)

_SYSTEM = (
    "You are a credit-risk analyst assistant. You write a short, clear, neutral "
    "explanation of an AUTOMATED credit decision for a loan officer. "
    "STRICT RULES:\n"
    "1. The decision is already final and was made by a statistical scorecard — "
    "never question, change, or second-guess it.\n"
    "2. Use ONLY the reason codes provided. Do NOT invent, assume, or add any "
    "factor that is not in the list.\n"
    "3. Be concise: 2–4 sentences, professional, plain English. No bullet lists, "
    "no markdown, no greetings.\n"
    "4. Do not give financial advice or promises."
)


def _format_reasons(result) -> str:
    def fmt(items):
        return "; ".join(
            f"{r['label']} = '{r['value']}'" for r in items
        ) or "none"

    return (
        f"Decision: {result.decision}\n"
        f"Probability of default: {result.probability_default:.1%}\n"
        f"Credit score: {result.credit_score} (grade {result.risk_grade})\n"
        f"Factors INCREASING risk: {fmt(result.reasons_increasing_risk)}\n"
        f"Factors LOWERING risk: {fmt(result.reasons_lowering_risk)}"
    )


def explain(result) -> str | None:
    """Generate a grounded narrative for an InferenceResult, or None on failure."""
    if not enabled:
        return None

    prompt = (
        f"{_format_reasons(result)}\n\n"
        "Write the explanation now, grounded only in the factors above."
    )
    try:
        resp = httpx.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": settings.ollama_model,
                "system": _SYSTEM,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=settings.ollama_timeout_s,
        )
        resp.raise_for_status()
        text = (resp.json().get("response") or "").strip()
        return text or None
    except Exception as exc:  # pragma: no cover - infra dependent
        log.warning("LLM explanation unavailable: %s", exc)
        return None
