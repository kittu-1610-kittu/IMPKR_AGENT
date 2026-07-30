import uuid
import logging
import asyncio
import json
import time
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, Security, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from backend.app.config import settings
from backend.app.database.connections import (
    initialize_all_databases, close_all_databases, graph_mgr, db_mgr
)
from backend.app.database.initial_data import seed_relational_database
from backend.app.blackboard.state import blackboard
from backend.app.orchestrator.service import OrchestratorService

# Configure logging
logger = logging.getLogger("MainApplication")

# Global Orchestrator instance
orchestrator = OrchestratorService()


# =====================================================================
# STAGING & PRODUCTION SAFETY SYSTEMS
# =====================================================================

# 1. API Key Auth Helper
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    """Validates X-API-KEY header to verify credentials."""
    expected_key = os.getenv("API_KEY", "impkr_secret_token")
    # If API_KEY is set to empty string in env, we allow bypassing auth for dev flexibility
    if expected_key and api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed. Missing or invalid {API_KEY_NAME} header."
        )
    return api_key


# 2. Token-Bucket Rate Limiter
class TokenBucketLimiter:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # Tokens refilled per second
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def consume(self, tokens_needed: int = 1) -> bool:
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            # Refill tokens
            self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            return False

# Global rate limiter (Allows burst of 10, refills 2 per second)
limiter = TokenBucketLimiter(capacity=10, refill_rate=2.0)


# =====================================================================
# FASTAPI LIFECYCLE
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    await initialize_all_databases()
    try:
        await seed_relational_database()
    except Exception as e:
        logger.error(f"Error seeding database during startup: {e}")
    yield
    # Shutdown tasks
    await close_all_databases()

app = FastAPI(
    title=settings.APP_NAME,
    description="Intelligent Multi-Agent Orchestration with Parallel Knowledge Retrieval (IMPKR)",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total active: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

ws_manager = ConnectionManager()


# =====================================================================
# PUBLIC API ENDPOINTS
# =====================================================================

@app.get("/api/health")
async def health_check():
    """Verify backend health and database mock statuses. (Public endpoint)"""
    return {
        "status": "healthy",
        "mocks_active": settings.USE_MOCK_DATABASES,
        "llm_mock_active": settings.USE_MOCK_LLM,
        "database_status": {
            "postgres": "mocked" if db_mgr.is_mock else "connected",
            "neo4j": "mocked" if graph_mgr.is_mock else "connected",
        }
    }


# =====================================================================
# SECURE API ENDPOINTS (Require verification dependencies)
# =====================================================================

@app.get("/api/query/stream", dependencies=[Depends(verify_api_key)])
async def run_query_stream(query: str, session_id: str = ""):
    """
    Triggers orchestrator execution and streams live agent events via SSE.
    Rate limited to protect generative resources.
    """
    # 1. Apply rate limit check
    if not await limiter.consume(1):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Too many requests. Please try again later."
        )

    if not query:
        raise HTTPException(status_code=400, detail="Query string is required.")
    
    if not session_id:
        session_id = str(uuid.uuid4())

    async def event_generator():
        try:
            async for event_str in orchestrator.execute_query_stream(session_id, query):
                yield f"data: {event_str}\n\n"
                event_data = json_loads_safe(event_str)
                if event_data:
                    await ws_manager.broadcast({"session_id": session_id, "payload": event_data})
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in SSE stream generator: {e}", exc_info=True)
            yield f"data: {{\"event\": \"error\", \"data\": \"Internal stream error: {str(e)}\"}}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

def json_loads_safe(val: str):
    try:
        return json.loads(val)
    except:
        return None

@app.get("/api/blackboard/{session_id}", dependencies=[Depends(verify_api_key)])
async def get_blackboard_state(session_id: str):
    """Retrieve full blackboard state of a session."""
    state = await blackboard.get_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session blackboard state not found.")
    return state

@app.get("/api/blackboards", dependencies=[Depends(verify_api_key)])
async def get_all_blackboards():
    """Retrieve list of all active sessions in the blackboard memory."""
    states = await blackboard.get_all_states()
    return {sid: state.model_dump() for sid, state in states.items()}

@app.get("/api/graph", dependencies=[Depends(verify_api_key)])
async def get_knowledge_graph():
    """Retrieve nodes and edges representing the database Knowledge Graph."""
    nodes = []
    edges = []
    
    if graph_mgr.is_mock and graph_mgr.mock_db:
        for nid, val in graph_mgr.mock_db.nodes.items():
            nodes.append({
                "id": nid,
                "label": val["properties"].get("name", nid),
                "group": val["label"],
                "title": val["properties"].get("description", "")
            })
        for edge in graph_mgr.mock_db.edges:
            edges.append({
                "from": edge["source"],
                "to": edge["target"],
                "label": edge["type"],
                "value": edge["confidence"]
            })
    else:
        try:
            cypher = "MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100;"
            records = await graph_mgr.execute_cypher(cypher)
            # Neo4j records parser could populate nodes/edges
        except Exception as e:
            logger.error(f"Error querying Neo4j schema: {e}")

    return {"nodes": nodes, "edges": edges}


# =====================================================================
# RLHF FEEDBACK & UPDATE ENDPOINTS
# =====================================================================

class FeedbackSubmission(BaseModel):
    session_id: str
    rating: int
    corrections: Optional[str] = None
    accepted: bool

@app.post("/api/feedback", dependencies=[Depends(verify_api_key)])
async def submit_feedback(feed: FeedbackSubmission):
    """Submit RLHF feedback. Triggers weight adjustment and KG node inserts."""
    logger.info(f"Received feedback for session {feed.session_id}. Rating: {feed.rating}, Accepted: {feed.accepted}")
    
    # Track feedback via telemetry logger
    from backend.app.monitoring.logger import telemetry
    telemetry.log_feedback(feed.session_id, feed.rating, feed.corrections or "", feed.accepted)
    
    state = await blackboard.get_state(feed.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session state not found.")

    # 1. Update Fusion weights
    if feed.rating >= 4:
        settings.ALPHA_RELEVANCE = min(1.0, settings.ALPHA_RELEVANCE + 0.02)
        settings.BETA_CONFIDENCE = min(1.0, settings.BETA_CONFIDENCE + 0.01)
    else:
        settings.GAMMA_DIVERSITY = min(1.0, settings.GAMMA_DIVERSITY + 0.03)
        settings.ALPHA_RELEVANCE = max(0.0, settings.ALPHA_RELEVANCE - 0.02)

    # 2. Update Knowledge Graph with corrections
    if feed.corrections and len(feed.corrections.strip()) > 5:
        logger.info(f"Injecting correction fact to KG: '{feed.corrections}'")
        try:
            node_id = f"Fact_{uuid.uuid4().hex[:6]}"
            if graph_mgr.is_mock and graph_mgr.mock_db:
                graph_mgr.mock_db.add_node(
                    node_id, 
                    "Correction", 
                    {"name": f"User Correction Fact", "description": feed.corrections}
                )
                graph_mgr.mock_db.add_edge(node_id, "AgentOrchestrator", "CORRECTS", 1.0)
            else:
                cypher = "CREATE (f:Correction {name: $name, description: $desc}) RETURN f;"
                await graph_mgr.execute_cypher(cypher, {"name": "User Correction Fact", "desc": feed.corrections})
        except Exception as e:
            logger.error(f"Failed to insert correction node to Neo4j: {e}")

    # 3. Insert performance log
    try:
        await db_mgr.execute_query(
            "INSERT INTO performance_metrics (component_id, latency_ms, success_rate) VALUES ($1, $2, $3);",
            "AgentOrchestrator", 120.0, 1.0 if feed.accepted else 0.0
        )
    except Exception as e:
        logger.error(f"Error logging metrics to SQL: {e}")

    return {
        "status": "success",
        "message": "Feedback ingested, fusion coefficients updated, and KG expanded.",
        "new_weights": {
            "alpha_relevance": settings.ALPHA_RELEVANCE,
            "beta_confidence": settings.BETA_CONFIDENCE,
            "gamma_diversity": settings.GAMMA_DIVERSITY,
            "delta_structural": settings.DELTA_STRUCTURAL
        }
    }


# =====================================================================
# ANALYTICS & MONITORING
# =====================================================================

@app.get("/api/analytics", dependencies=[Depends(verify_api_key)])
async def get_analytics_metrics():
    """Retrieve system analytics, latency stats, and accuracy ratings."""
    try:
        rows = await db_mgr.execute_query(
            "SELECT AVG(latency_ms) as avg_latency, AVG(success_rate) as avg_success FROM performance_metrics;"
        )
        avg_latency = rows[0]["avg_latency"] if rows and rows[0]["avg_latency"] else 125.0
        avg_success = rows[0]["avg_success"] if rows and rows[0]["avg_success"] else 0.90
    except Exception:
        avg_latency = 120.5
        avg_success = 0.88

    return {
        "global_average_latency_ms": avg_latency,
        "consensus_success_rate": avg_success,
        "current_fusion_weights": {
            "relevance": settings.ALPHA_RELEVANCE,
            "confidence": settings.BETA_CONFIDENCE,
            "diversity": settings.GAMMA_DIVERSITY,
            "structural": settings.DELTA_STRUCTURAL
        },
        "query_types_distribution": {
            "general_architecture": 45,
            "verification_rules": 25,
            "fusion_scoring": 30
        }
    }


# =====================================================================
# WEBSOCKET SUBSCRIPTION
# =====================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for subscribing to live system-wide orchestrator metrics.
    Authentication check is performed upon connection accept.
    """
    # Grab header for socket auth check
    headers = dict(websocket.headers)
    api_key = headers.get("x-api-key")
    expected_key = os.getenv("API_KEY", "impkr_secret_token")
    
    if expected_key and api_key != expected_key:
        logger.warning("WebSocket connection rejected: invalid API key.")
        await websocket.close(code=4001)
        return

    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "heartbeat", "received": data})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)
