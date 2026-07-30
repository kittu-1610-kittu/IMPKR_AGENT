import asyncio
import time
import pytest
import numpy as np
from shared.schema import Evidence, ValidationResult
from backend.app.retrieval.web import WebRetriever, clean_html, optimize_query, breakers
from backend.app.blackboard.state import blackboard
from backend.app.evidence_fusion.synchronizer import EvidenceSynchronizer
from backend.app.evidence_fusion.fusion import EvidenceFuser
from backend.app.orchestrator.service import OrchestratorService
from backend.app.database.connections import initialize_all_databases, close_all_databases, redis_mgr
from backend.app.config import settings

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

def test_html_cleaning():
    raw_html = "<script>alert('spam');</script><div>Trusted content <style>body {}</style>from GitHub!</div>"
    cleaned = clean_html(raw_html)
    assert "alert" not in cleaned
    assert "GitHub" in cleaned
    assert "<" not in cleaned

def test_query_optimization():
    q = "What is the parallel latency formula in IMPKR-AGENT?"
    opt = optimize_query(q)
    assert "What" not in opt
    assert "is" not in opt
    assert "parallel latency formula" in opt.lower()

@pytest.mark.asyncio
async def test_web_retriever_caching_and_providers():
    await initialize_all_databases()
    retriever = WebRetriever()
    
    # Enable web retrieval and ensure mock databases
    settings.ENABLE_WEB_RETRIEVAL = True
    
    # 1. Clear cache
    opt_q = optimize_query("parallel latency")
    cache_key = f"web_results:{opt_q}"
    if redis_mgr.client:
        await redis_mgr.client.delete(cache_key)

    # 2. First query (triggers providers fetch and caching)
    results = await retriever.retrieve("parallel latency", "session_cache_test", limit=2)
    assert len(results) > 0
    assert results[0].source_type == "web"
    assert results[0].url is not None
    assert results[0].domain is not None
    assert results[0].title is not None
    assert len(results[0].embedding) == 768

    # 3. Second query (must be cache hit)
    results2 = await retriever.retrieve("parallel latency", "session_cache_test_hit", limit=2)
    assert len(results2) > 0
    assert results2[0].metadata.get("cache_hit") is True
    
    await close_all_databases()

@pytest.mark.asyncio
async def test_provider_fallback_and_circuit_breakers():
    # Force Tavily API call failure by setting an invalid key and using tavily provider
    settings.TAVILY_API_KEY = "invalid_key_to_force_failure"
    settings.WEB_PROVIDER = "tavily"
    
    retriever = WebRetriever()
    
    # Reset tavily breaker state
    breakers["tavily"].failures = 0
    breakers["tavily"].state = "CLOSED"

    # Execute search - Tavily will fail and fall back to Serper/DuckDuckGo or Mock
    results = await retriever.retrieve("graph validation", "session_breaker_test", limit=3)
    assert len(results) > 0
    
    # Assert tavily breaker recorded at least one failure
    assert breakers["tavily"].failures > 0

    # Restore key
    settings.TAVILY_API_KEY = ""

@pytest.mark.asyncio
async def test_parallel_execution_timings_and_blackboard():
    await initialize_all_databases()
    orc = OrchestratorService()
    
    # Execute query
    session_id = "web_session_parallel_test"
    final_resp = None
    
    async for event_str in orc.execute_query_stream(session_id, "Explain IMPKR-AGENT web search integration"):
        import json
        ev = json.loads(event_str)
        if ev.get("event") == "final_response":
            final_resp = ev.get("data")
            
    assert final_resp is not None
    
    # Verify Blackboard contains web integration fields
    state = await blackboard.get_state(session_id)
    assert state is not None
    assert state.web_results is not None
    assert len(state.web_results) > 0
    assert state.web_confidence > 0.0
    assert state.web_latency >= 0.0
    assert len(state.web_sources) > 0
    
    await close_all_databases()

@pytest.mark.asyncio
async def test_synchronizer_and_fuser_incorporation():
    await initialize_all_databases()
    
    raw_evs = [
        Evidence(id="e1", content="Component synchronizer", source_type="vector", confidence=0.8),
        Evidence(
            id="e2", 
            content="Production-Grade Web Retrieval Agent handles Tavily search queries.", 
            source_type="web", 
            confidence=0.9,
            title="Web Docs",
            url="https://github.com/docs/web",
            domain="github.com",
            snippet="Web Agent search",
            timestamp=time.time()
        )
    ]
    
    # Sync must deduplicate and copy web attributes
    synced = EvidenceSynchronizer.synchronize(raw_evs)
    web_synced = [e for e in synced if e.source_type == "web"]
    assert len(web_synced) == 1
    assert web_synced[0].title == "Web Docs"
    assert web_synced[0].url == "https://github.com/docs/web"
    
    # Fuse must incorporate web and calculate score
    fused = await EvidenceFuser.fuse("Web Search Retrieval", synced)
    assert len(fused) > 0
    
    await close_all_databases()

@pytest.mark.asyncio
async def test_validator_and_trust_agent_web_calculations():
    from backend.app.validator.agent import ValidatorAgent
    from backend.app.trust.agent import TrustAgent
    
    # 1. Validator
    val_agent = ValidatorAgent()
    fused = [
        Evidence(
            id="web_ev_1", 
            content="IMPKR-AGENT parallel latency holds max(Tvector, Tgraph, Tsql, Tweb).", 
            source_type="web",
            domain="github.com",
            url="https://github.com/docs"
        )
    ]
    report = await val_agent.validate("The web latency is parallelized.", fused)
    assert len(report) > 0
    
    # 2. Trust Agent
    trust_agent = TrustAgent()
    assessment = await trust_agent.assess_trust(
        validation_results=[
            ValidationResult(claim="latency formula", status="verified", evidence_id="web_ev_1", reasoning="matches docs", confidence_score=0.95)
        ],
        iteration=1
    )
    
    assert assessment["confidence"] > 0.0
    assert assessment["consensus_score"] > 0.0
    assert assessment["converged"] is not None
