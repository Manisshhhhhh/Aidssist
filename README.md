# Aidssist

## Structure

```text
Aidssist/
├── backend/
│   ├── aidssist_runtime/
│   ├── deploy/
│   ├── load_tests/
│   ├── scripts/
│   ├── tests/
│   ├── prompt_pipeline.py
│   ├── insight_engine.py
│   ├── dashboard_helpers.py
│   ├── data_quality.py
│   ├── data_sources.py
│   ├── chart_customization.py
│   └── workflow_store.py
├── frontend/
│   ├── app.py
│   └── assets/
├── web/
│   ├── src/
│   ├── package.json
│   └── Dockerfile
├── requirements.txt
├── docker-compose.yml
└── Dockerfile
```

## Run Locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Aidssist now expects Python 3.11+ locally. If you already have a `.venv` created with Python 3.9, remove or rename it first and recreate it with Python 3.11 before installing requirements.

## Run API Runtime

```bash
uvicorn backend.aidssist_runtime.api:app --reload
```

## Run React SaaS Frontend

```bash
cd web
npm install
cp .env.example .env
npm run dev
```

By default the React app uses `/api`.
In Vite dev, `/api` is proxied to `http://127.0.0.1:8000`.
In same-origin deployments, `/api` works through the bundled Nginx proxy.

The default upload limit is `500 MB`. You can override it with `AIDSSIST_MAX_UPLOAD_MB`.

## Public Demo

The backend now exposes a public demo endpoint:

```bash
GET /demo-data
```

It returns:

- bundled sample dataset metadata + preview rows
- precomputed analysis, forecast, and root-cause outputs
- dashboard KPI/chart payloads
- demo stats, guided steps, and suggested actions

`GET /demo` remains available as a compatibility alias.

## Backend Deployment

Render:

- `render.yaml` is included at the repo root.
- build command: `pip install --upgrade pip && pip install -r requirements.txt`
- start command: `uvicorn backend.aidssist_runtime.api:app --host 0.0.0.0 --port $PORT`
- health check: `/readyz`

Railway:

- `railway.json` is included at the repo root.
- root `Dockerfile` now has a default `CMD` for the FastAPI runtime.
- health check: `/readyz`

Recommended backend env vars for a public demo:

```bash
AIDSSIST_ENV=production
AIDSSIST_CORS_ORIGINS=https://your-frontend.vercel.app
AIDSSIST_API_URL=https://your-backend.onrender.com
AIDSSIST_OBJECT_STORE_BACKEND=local
AIDSSIST_REDIS_URL=
GEMINI_API_KEY=your_key_if_you_want_live_model_features
```

If `AIDSSIST_DATABASE_URL` is omitted, Aidssist falls back to a local SQLite database inside `.aidssist/`.

## Frontend Deployment

Vercel:

- `web/vercel.json` is included for SPA rewrites.
- set the Vercel project root to `web/`
- configure `VITE_AIDSSIST_API_URL` to your deployed backend URL

Example:

```bash
VITE_AIDSSIST_API_URL=https://your-backend.onrender.com
```

## SaaS API Additions

- `POST /v1/auth/register`
- `POST /v1/auth/login`
- `GET /v1/auth/me`
- `POST /v1/auth/logout`
- `GET /v1/datasets`
- `GET /v1/datasets/{dataset_id}`
- `GET /v1/history`

Authenticated uploads and analysis jobs are now linked to the signed-in user so the React client can show reusable dataset state and analysis history.

## Run Tests

```bash
./venv/bin/python -m unittest discover -s backend/tests -t .
```
