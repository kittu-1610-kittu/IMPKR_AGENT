# Installation Guide

Follow these steps to set up the IMPKR-AGENT workspace entirely from your VS Code terminal.

---

## 📋 System Prerequisites

Ensure you have Python 3.10+ installed on your system.

---

## 🛠 Step 1: Install Dependencies

Open a terminal at the project root (`e:\hari main project\IMPKR - AGENT`) and install the python packages:

```bash
pip install -r backend/requirements.txt
```

*(Note: Dependencies include fastapi, uvicorn, pydantic-settings, httpx, numpy, faiss-cpu, openai, google-generativeai, and asyncpg/neo4j driver hooks).*

---

## 🔑 Step 2: Configure Environment Variables

Create or set the following environment variables in your terminal, or define them in a `.env` file at the root:

```ini
# Core LLM API Key (For Gemini/Google GenAI)
GEMINI_API_KEY=your_gemini_api_key_here

# API Security Headers (Default matched by CLI client)
API_KEY=impkr_secret_token

# Fallbacks for Local Mock Runs
USE_MOCK_LLM=False
USE_MOCK_DATABASES=True

# Connection Strings (If USE_MOCK_DATABASES is False)
POSTGRES_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/impkr_agent
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
REDIS_URL=redis://localhost:6379/0
```

---

## 🗄 Step 3: Run Database Stack (Optional)

If running in production mode (`USE_MOCK_DATABASES=False`), spin up the database containers:

```bash
cd deployment
docker-compose up -d
```

This starts:
- **PostgreSQL** on port `5432`
- **Neo4j Community** on port `7687` (Bolt) and `7474` (HTTP)
- **Redis Cache** on port `6379`
