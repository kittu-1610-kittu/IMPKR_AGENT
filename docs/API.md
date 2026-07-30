# Backend API Reference Guide

The FastAPI backend exposes endpoints for streaming execution loops, posting human feedback, and reading system metrics.

---

## 🔑 Authentication

All core API endpoints require the inclusion of an API key in request headers:

```http
X-API-KEY: impkr_secret_token
```

*Requests with missing or invalid keys return a `401 Unauthorized` response.*

---

## 🚦 Rate Limiting

Endpoints are protected by a token-bucket rate limiter. 
- **Capacity**: 10 tokens.
- **Refill Rate**: 2.0 tokens/second.
- Flooding the server with concurrent requests drains the bucket, yielding a `429 Too Many Requests` status code.

---

## 📡 Endpoints List

### 1. SSE Query Stream
- **Path**: `/api/query/stream`
- **Method**: `GET`
- **Parameters**: 
  - `query` (str, required): Search query.
  - `session_id` (str, optional): Custom execution tracking ID.
- **Response**: Server-Sent Events (SSE) data streaming execution steps, plans, evidence, agent critiques, and the final answer.

### 2. Post Feedback (RLHF Loop)
- **Path**: `/api/feedback`
- **Method**: `POST`
- **Body JSON**:
  ```json
  {
    "session_id": "cli_uuid_here",
    "rating": 5,
    "corrections": "Optional corrections context to update Neo4j KG relationships",
    "accepted": true
  }
  ```
- **Response**: Returns the updated adaptive fusion weight coefficients.

### 3. Server Health
- **Path**: `/api/health`
- **Method**: `GET`
- **Response**: `{"status": "ok", "db_connected": true}`. (Does not require X-API-KEY).

### 4. Telemetry Metrics
- **Path**: `/api/metrics`
- **Method**: `GET`
- **Response**: Overall tracking metrics (processed query counts, mean latency, convergence rates).

---

## 📡 Blackboard Schema Web Fields
When retrieving session state (via `execute_query_stream` state events or internal blackboard endpoints), the returned session data contains the following integration properties:

```json
{
  "session_id": "cli_abc123",
  "query": "Explain parallel retrieval",
  "status": "done",
  "web_results": [
    {
      "id": "web_cli_abc123_0",
      "content": "...",
      "source_type": "web",
      "confidence": 0.82,
      "title": "IMPKR-AGENT Architecture Details",
      "url": "https://github.com/docs",
      "domain": "github.com",
      "snippet": "...",
      "retrieval_time": 420.5,
      "timestamp": 1785239482.0
    }
  ],
  "web_confidence": 0.82,
  "web_latency": 420.5,
  "web_sources": [
    "https://github.com/docs"
  ]
}
```
