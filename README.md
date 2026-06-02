# Credit Risk Platform — MVP

Production-style deployment of the CAMPARI credit-scoring model (Weight-of-Evidence
encoding + Logistic Regression, trained on the Statlog German Credit dataset).
FastAPI serves both a JSON API and a fintech-style web UI; predictions are stored
in PostgreSQL and logged to MLflow. The model is **reused as-is** — no retraining.

---

## 1. Inference Contract

**Input** — 20 raw German-credit fields (validated by `app/models/schemas.py`):

| Group | Fields |
|-------|--------|
| Numeric | `duration`, `credit_amount`, `installment_rate`, `residence`, `age`, `credit_cards`, `dependents` |
| Categorical (A-codes) | `account_status`, `credit_history`, `purpose`, `savings`, `employment`, `personal_status`, `guarantors`, `property`, `other_installments`, `housing`, `job`, `phone`, `foreign_worker` |

Numeric bounds come from the notebook's credit-policy asserts; categoricals must be
the documented `A##` codes (see `ml/reference.py`).

**Preprocessing** (`ml/preprocess.py`) — reproduces the notebook exactly:

1. Assemble the 20 raw columns in training order.
2. `scorecardpy.woebin_ply(X, woe_bins)` → applies the **trained** WoE bins
   (`ml/woe_bins.pkl`). No refitting.
3. Select the **16** model features from `ml/feature_columns.pkl` (4 dropped in the
   notebook for IV < 0.01), in the trained order.

**Model expectation** — `sklearn LogisticRegression` (`ml/model.pkl`, 16 WoE inputs,
`classes_ = [0, 1]`). `predict_proba(X)[:, 1] = P(Good)`; `P(default) = 1 − P(Good)`.

**Decision engine** (`ml/inference.py`, from notebook cell 58 + `scoring_config.pkl`,
`best_threshold=0.58`, `buffer=0.05`):

| P(Good) | Decision |
|---------|----------|
| ≥ 0.63 | **Approve** |
| ≤ 0.53 | **Reject** |
| otherwise | **Manual Review** |

**Score** — scorecard scaling reconstructed from `scoring_config.pkl`
(`BASE_SCORE=600`, `PDO=50`): `score = 600 + (50/ln2)·ln(odds_good)`, clamped to
300–850. Display only; never affects the decision. Risk grade A–F is derived from
P(default).

> **Parity is enforced by tests** — `tests/test_inference_parity.py` rebuilds the
> notebook's train/test split and asserts the engine's `P(Good)` matches the raw
> notebook path row-by-row (`abs=1e-9`) and reproduces the notebook AUC (~0.767).

---

## 2. Architecture

```
Browser ──► FastAPI (UI routes, Jinja2 + Tailwind)
                │
HTTP client ──► FastAPI (JSON API: /api/*)
                │
                ├── ml/      inference pipeline (WoE + LogReg, reused .pkl)
                ├── PostgreSQL  applications + predictions  (system of record)
                └── MLflow    inference tracking (latency, P(default), score)
```

All three services (`api`, `postgres`, `mlflow`) run via Docker Compose. The `api`
and `mlflow` services share one image; `mlflow` just runs a different command.

---

## 3. Project Structure

The deployable service lives at the platform root, beside the original `Data/`
and `Notebook/` folders.

```
Credit Risk Platform/
├── app/
│   ├── api/routes.py          # JSON API: /api/predict, /api/applications, /api/health
│   ├── ui/routes.py           # Web pages: /, /predict, /applications, /health
│   ├── services/              # prediction orchestration (inference + DB + MLflow)
│   ├── models/schemas.py      # Pydantic request/response models
│   ├── db/                    # SQLAlchemy engine + ORM models
│   ├── core/config.py         # env-driven settings
│   └── main.py                # FastAPI app (mounts API + UI)
├── ml/
│   ├── model.pkl, woe_bins.pkl, feature_columns.pkl, scoring_config.pkl
│   ├── reference.py           # raw schema + field labels (single source of truth)
│   ├── preprocess.py          # WoE apply + feature selection (notebook parity)
│   ├── inference.py           # probability → score → decision engine
│   └── mlflow_logger.py       # best-effort inference tracking
├── templates/                 # index, result, history, health (Tailwind)
├── static/app.css
├── alembic/                   # versioned DB migrations
├── tests/                     # API, schema, notebook-parity (real Postgres)
├── Dockerfile  docker-compose.yml  requirements.txt
├── Data/                      # original German Credit dataset (used by parity test)
└── Notebook/                  # original training notebook
```

---

## 4. Run (Docker Compose — recommended)

```bash
docker-compose up --build
```

Then open:

| URL | What |
|-----|------|
| http://localhost:8000/           | Loan application form |
| http://localhost:8000/applications | Decision history |
| http://localhost:8000/health     | System status |
| http://localhost:8000/docs       | OpenAPI / Swagger |
| http://localhost:5000/           | MLflow UI |

Postgres is exposed on host port **5433** (container 5432) to avoid clashing with a
local Postgres.

## 5. Run locally (without Docker)

Requires a reachable PostgreSQL. Reuse the compose DB or set your own DSN:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env          # edit DATABASE_URL / MLFLOW_TRACKING_URI
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

## 6. Tests

The suite runs against a **real Postgres**: it starts an ephemeral
`postgres:16-alpine` container via testcontainers (Docker daemon required), or set
`TEST_DATABASE_URL` to use an existing instance.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Covers: API endpoints, schema validation, and notebook↔API inference parity.

---

## API quick reference

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/predict` | Score an application, persist it, return the decision |
| POST | `/api/applications` | Alias of `/api/predict` |
| GET  | `/api/applications` | List scored applications |
| GET  | `/api/applications/{id}` | Fetch one application + its prediction |
| GET  | `/api/health` | Model / DB / MLflow status |

```bash
curl -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" \
  -d '{"account_status":"A14","duration":12,"credit_history":"A32","purpose":"A43",
       "credit_amount":1500,"savings":"A65","employment":"A75","installment_rate":2,
       "personal_status":"A93","guarantors":"A101","residence":2,"property":"A121",
       "age":40,"other_installments":"A143","housing":"A152","credit_cards":1,
       "job":"A173","dependents":1,"phone":"A192","foreign_worker":"A202"}'
```
