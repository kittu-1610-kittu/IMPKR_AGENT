# Execution Guide

You can run the IMPKR-AGENT system in two modes entirely from your VS Code terminal.

---

## Mode 1: In-Process Direct CLI Execution (Fastest & Simplest)

In Direct Mode, the CLI manages database lifespans in-process. There is no need to launch the FastAPI server in a separate terminal.

Run the CLI script:
```bash
python cli.py
```

*The CLI detects that port 8000 is inactive, automatically initializes the SQLite fallback database, seeds tables, and lets you query the multi-agent pipeline immediately in your terminal.*

---

## Mode 2: Multi-Terminal API Mode (Production Setup)

In this mode, you boot the FastAPI web server, and the CLI acts as an authenticated client connecting to it over HTTP.

### Terminal A: Start the FastAPI Server
Open a terminal at the project root and run:
```bash
# Set environment token
$env:API_KEY="impkr_secret_token"

# Run Uvicorn
uvicorn backend.app.main:app --reload --port 8000
```
This boots the FastAPI application on `http://localhost:8000`.

### Terminal B: Start the CLI client
Open a second terminal at the project root and run:
```bash
python cli.py
```
*The CLI registers a successful handshake with the server and streams tokenized SSE query executions.*
