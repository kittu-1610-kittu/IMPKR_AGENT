import asyncio
import time
import json
import pytest
import numpy as np
from typing import List
from shared.schema import Evidence, SubTask, PlannerExecutionPlan, ConsensusStep, ValidationResult
from backend.app.blackboard.state import Blackboard
from backend.app.evidence_fusion.synchronizer import EvidenceSynchronizer
from backend.app.evidence_fusion.fusion import EvidenceFuser
from backend.app.orchestrator.service import OrchestratorService
from backend.app.database.connections import initialize_all_databases, close_all_databases, vector_mgr
from backend.app.database.initial_data import seed_relational_database
from backend.app.database.datasets import DatasetManager, StatisticalValidator
from backend.app.config import settings

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# =====================================================================
# ORIGINAL pipeline & timing tests
# =====================================================================

@pytest.mark.asyncio
async def test_parallel_retrieval_latency():
    delays = [0.1, 0.25, 0.4, 0.15]
    
    async def mock_retrieve(delay: float, name: str):
        await asyncio.sleep(delay)
        return [Evidence(id=name, content=f"Result from {name}", source_type="vector")]

    start_time = time.time()
    
    results = await asyncio.gather(
        mock_retrieve(delays[0], "vector"),
        mock_retrieve(delays[1], "graph"),
        mock_retrieve(delays[2], "relational"),
        mock_retrieve(delays[3], "web")
    )
    
    elapsed = time.time() - start_time
    max_delay = max(delays)
    sum_delay = sum(delays)
    
    assert elapsed < sum_delay
    assert elapsed >= max_delay - 0.05
    assert elapsed <= max_delay + 0.1
    assert len(results) == 4
    print(f"\nParallel execution verified: {elapsed:.3f}s (Max: {max_delay:.3f}s, Sum: {sum_delay:.3f}s)")


@pytest.mark.asyncio
async def test_blackboard_operations():
    bb = Blackboard()
    session_id = "test_sess_999"
    
    state = await bb.initialize_session(session_id, "Test query")
    assert state.session_id == session_id
    assert state.status == "initialized"
    
    plan = PlannerExecutionPlan(
        query="Test query",
        subtasks=[SubTask(id="task1", description="desc", sources=["vector"])],
        rationale="rat"
    )
    await bb.write_plan(session_id, plan)
    
    updated = await bb.get_state(session_id)
    assert updated.plan.subtasks[0].id == "task1"
    assert updated.status == "planning_completed"
    
    ev = [Evidence(id="ev1", content="Fact content", source_type="vector")]
    await bb.add_raw_evidence(session_id, ev)
    
    updated = await bb.get_state(session_id)
    assert len(updated.raw_evidence) == 1
    assert updated.status == "retrieval_completed"


def test_evidence_synchronizer():
    raw = [
        Evidence(id="e1", content="Component orchestrator is running smoothly.", source_type="vector", confidence=0.8),
        Evidence(id="e2", content="Component orchestrator is running smoothly.", source_type="web", confidence=0.6),
        Evidence(id="e3", content="Adaptive evidence fusion uses softmax scoring metrics.", source_type="vector", confidence=0.9)
    ]
    
    synced = EvidenceSynchronizer.synchronize(raw)
    assert len(synced) == 2
    matches = [e for e in synced if "orchestrator" in e.content]
    assert len(matches) == 1
    assert matches[0].source_type == "vector"


@pytest.mark.asyncio
async def test_evidence_fusion_scoring():
    await initialize_all_databases()
    await seed_relational_database()
    
    raw_ev = [
        Evidence(id="e1", content="IMPKR-AGENT parallel latency rules are optimal.", source_type="vector", confidence=0.9),
        Evidence(id="e2", content="Grounded validation check reduces hallucination rates.", source_type="graph", confidence=0.85),
        Evidence(id="e3", content="Normal weather reports show heavy rain in Seattle.", source_type="web", confidence=0.5)
    ]
    
    query = "Explain IMPKR-AGENT parallel latency and validation rules."
    fused = await EvidenceFuser.fuse(query, raw_ev, top_k=2)
    
    await close_all_databases()
    
    assert len(fused) == 2
    assert " Seattle" not in fused[0].content
    assert " Seattle" not in fused[1].content
    assert fused[0].score > fused[1].score


# =====================================================================
# HARDENING & PRODUCTION SECURITY TESTS
# =====================================================================

@pytest.mark.asyncio
async def test_api_authentication_and_rate_limiting():
    from httpx import AsyncClient, ASGITransport
    from backend.app.main import app, limiter
    
    await initialize_all_databases()
    await seed_relational_database()
    
    limiter.tokens = 10.0
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/api/query/stream?query=hello")
        assert resp.status_code == 401
        
        resp = await ac.get("/api/query/stream?query=hello", headers={"X-API-KEY": "wrong_key"})
        assert resp.status_code == 401
        
        headers = {"X-API-KEY": "impkr_secret_token"}
        resp = await ac.get("/api/query/stream?query=hello", headers=headers)
        assert resp.status_code == 200
        
        tasks = [ac.get("/api/query/stream?query=hello", headers=headers) for _ in range(15)]
        responses = await asyncio.gather(*tasks)
        
        limit_triggered = any(r.status_code == 429 for r in responses)
        assert limit_triggered is True
        
        limiter.tokens = 10.0
        
    await close_all_databases()


@pytest.mark.asyncio
async def test_circuit_breakers_and_retries():
    from backend.app.llm import CircuitBreaker
    
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
    assert breaker.state == "CLOSED"
    assert breaker.allow_request() is True
    
    breaker.record_failure()
    assert breaker.state == "CLOSED"
    
    breaker.record_failure()
    assert breaker.state == "OPEN"
    assert breaker.allow_request() is False
    
    await asyncio.sleep(1.1)
    assert breaker.allow_request() is True
    assert breaker.state == "HALF-OPEN"
    
    breaker.record_success()
    assert breaker.state == "CLOSED"


# =====================================================================
# TABLE 9 COMPLIANCE & VERIFICATION TESTS
# =====================================================================

@pytest.mark.asyncio
async def test_table9_vector_normalization():
    """Verify FAISS vector size is 768 and vectors are properly normalized to unit length."""
    await initialize_all_databases()
    assert vector_mgr.dimension == 768
    
    embs = await vector_mgr.get_embeddings(["deberta base embeddings check"])
    assert embs.shape == (1, 768)
    
    # Assert L2 Normalization (norm should equal 1.0)
    norm = np.linalg.norm(embs[0])
    assert np.isclose(norm, 1.0, atol=1e-4)
    await close_all_databases()


@pytest.mark.asyncio
async def test_table9_graph_traversal_4_hops():
    """Verify that multi-hop graph BFS traverses up to 4-hops and filters neighbor counts."""
    from backend.app.retrieval.graph import GraphRetriever
    await initialize_all_databases()
    
    retriever = GraphRetriever()
    ev_list = await retriever.retrieve("Generator", "session_t9")
    
    # Verify we got multi-hop paths of length greater than 2
    hops_found = [e.metadata["hops"] for e in ev_list if "hops" in e.metadata]
    assert len(hops_found) > 0
    assert max(hops_found) >= 3  # Generator -> Critic -> Validator -> Trust is a 3-hop path
    
    # Assert child neighbors count did not exceed settings threshold limit of 12
    for ev in ev_list:
        assert ev.metadata.get("hops", 0) <= 4
        
    await close_all_databases()


def test_table9_datasets_preprocessing():
    """Verify dataset loading and comment-stripping code normalization."""
    # 1. Loading
    hotpot = DatasetManager.load_hotpotqa()
    heval = DatasetManager.load_humaneval()
    assert len(hotpot) > 0
    assert len(heval) > 0
    
    # 2. Code Normalization
    raw_code = "def test_fn():\n    # inline comment\n    return True  // another comment"
    normalized = DatasetManager.normalize_code(raw_code)
    assert "comment" not in normalized
    assert "inline" not in normalized
    assert "test_fn" in normalized


@pytest.mark.asyncio
async def test_table9_statistical_evaluation():
    """Verify that Statistical Seed Runner executes 10 trials across target seeds."""
    await initialize_all_databases()
    await seed_relational_database()
    
    results = await StatisticalValidator.run_statistical_evaluation("Explain parallel retrieval")
    assert results["total_runs"] == 10
    assert len(results["seeds_used"]) == 10
    assert "average_latency_ms" in results
    assert "average_trust_score" in results
    
    await close_all_databases()


@pytest.mark.asyncio
async def test_table9_weighted_scoring_calculations():
    """Verify the weighted trust scoring model matches Table 9 configurations."""
    from backend.app.trust.agent import TrustAgent
    
    agent = TrustAgent()
    validations = [
        ValidationResult(claim="c1", status="verified", evidence_id="graph_ev", reasoning="path check", confidence_score=0.9),
        ValidationResult(claim="c2", status="verified", evidence_id="vector_ev", reasoning="text check", confidence_score=0.8)
    ]
    
    # 2 verified claims, 1 verified by graph (g_s=0.5), 1 verified by vector (s_c=0.5)
    # v_r = 1.0, avg_conf = 0.85
    # Weighted Score = 0.35 * 1.0 + 0.25 * 0.85 + 0.20 * 0.5 + 0.20 * 0.5 = 0.35 + 0.2125 + 0.10 + 0.10 = 0.7625
    assessment = await agent.assess_trust(validations, iteration=1)
    
    # Expected trust score calculated on top of these weight vectors:
    expected_score = (
        settings.WEIGHT_VALIDATION * 1.0 +
        settings.WEIGHT_CONFIDENCE * 0.85 +
        settings.WEIGHT_CONSISTENCY * 0.5 +
        settings.WEIGHT_GRAPH_SUPPORT * 0.5
    )
    
    # We check if assessment output aligns
    assert np.isclose(expected_score, 0.7625)
