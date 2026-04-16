# Aidssist

Aidssist is an AI-assisted analytics platform for turning raw business data into guided insights, forecast outputs, and operational recommendations. The repository combines a Streamlit workflow application, a FastAPI runtime, and a React/Vite web client in one deployment-ready codebase.

## Highlights

- AI-assisted dataset understanding, workflow guidance, and result generation
- Streamlit application for interactive analysis flows and operational dashboards
- FastAPI runtime for authenticated analysis, datasets, history, and demo endpoints
- React/Vite client for a SaaS-style frontend experience
- Docker and cloud deployment configuration for local development and hosted environments

## Tech Stack

- Python 3.11+
- Streamlit
- FastAPI and Uvicorn
- React, TypeScript, and Vite
- Docker and Docker Compose

## Repository Structure

```text
Aidssist/
├── backend/    Core analytics, API runtime, services, and tests
├── frontend/   Streamlit UI and local app assets
├── web/        React/Vite frontend
├── app.py      Root launcher for the Streamlit app
└── requirements.txt
```

## Quick Start

### 1. Clone the repository

```bash
git clone git@github.com:Manisshhhhhh/Aidssist.git
cd Aidssist
```

### 2. Run the Streamlit application

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### 3. Run the FastAPI runtime

```bash
source .venv/bin/activate
uvicorn backend.aidssist_runtime.api:app --reload
```

### 4. Run the React web client

```bash
cd web
npm install
npm run dev
```

## What Recruiters and Reviewers Can Expect

- A modular Python backend with dedicated services, runtime configuration, and test coverage
- Separate product surfaces for Streamlit workflows and a web-based frontend
- Local and cloud deployment support through Docker, Render, and Railway configuration
- A codebase structured for extending AI workflows, dataset ingestion, forecasting, and dashboard delivery

## Testing

```bash
python -m unittest discover -s backend/tests -t .
```

## Deployment Notes

- `Dockerfile` and `docker-compose.yml` are included for containerized environments
- `render.yaml` and `railway.json` are included for hosted backend deployment
- The React app lives under `web/` and can be deployed independently from the Python runtime

## Future Improvements

- Expand authentication and user management flows
- Improve frontend polish and reporting workflows
- Add more production observability and CI automation
