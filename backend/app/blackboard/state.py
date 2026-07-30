import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from shared.schema import AgentState, PlannerExecutionPlan, Evidence, ConsensusStep
from backend.app.database.connections import redis_mgr

logger = logging.getLogger("Blackboard")

class Blackboard:
    def __init__(self):
        # In-memory dictionary backing
        self._memory_db: Dict[str, AgentState] = {}
        self._lock = asyncio.Lock()

    async def get_state(self, session_id: str) -> Optional[AgentState]:
        """Retrieve the state of a session from Redis (if enabled) or memory."""
        # Try Redis first if not mock
        if not redis_mgr.is_mock and redis_mgr.client:
            try:
                data = await redis_mgr.client.get(f"blackboard:{session_id}")
                if data:
                    return AgentState.model_validate_json(data)
            except Exception as e:
                logger.error(f"Failed to read from Redis ({e}). Using in-memory fallback.")
        
        async with self._lock:
            return self._memory_db.get(session_id)

    async def save_state(self, session_id: str, state: AgentState) -> None:
        """Persist the state to Redis (if enabled) and in-memory."""
        # Write to memory
        async with self._lock:
            self._memory_db[session_id] = state

        # Write to Redis
        if not redis_mgr.is_mock and redis_mgr.client:
            try:
                state_json = state.model_dump_json()
                await redis_mgr.client.set(f"blackboard:{session_id}", state_json)
            except Exception as e:
                logger.error(f"Failed to write to Redis ({e}).")

    async def initialize_session(self, session_id: str, query: str) -> AgentState:
        """Create a new session state on the blackboard."""
        state = AgentState(
            session_id=session_id,
            query=query,
            status="initialized"
        )
        await self.save_state(session_id, state)
        logger.info(f"Initialized blackboard session {session_id} for query: '{query}'")
        return state

    async def update_status(self, session_id: str, status: str) -> None:
        """Update session status."""
        state = await self.get_state(session_id)
        if state:
            state.status = status
            await self.save_state(session_id, state)
            logger.info(f"Session {session_id} status updated to: {status}")

    async def write_plan(self, session_id: str, plan: PlannerExecutionPlan) -> None:
        """Write the decomposed execution plan to the blackboard."""
        state = await self.get_state(session_id)
        if state:
            state.plan = plan
            state.status = "planning_completed"
            await self.save_state(session_id, state)
            logger.info(f"Session {session_id} planner execution plan saved.")

    async def add_raw_evidence(self, session_id: str, evidence: List[Evidence]) -> None:
        """Append newly retrieved raw evidence into the blackboard."""
        state = await self.get_state(session_id)
        if state:
            state.raw_evidence.extend(evidence)
            state.status = "retrieval_completed"
            await self.save_state(session_id, state)
            logger.info(f"Session {session_id} raw evidence updated. Count: {len(state.raw_evidence)}")

    async def write_fused_evidence(self, session_id: str, fused: List[Evidence]) -> None:
        """Save the adaptive evidence fusion results."""
        state = await self.get_state(session_id)
        if state:
            state.fused_evidence = fused
            state.status = "fusion_completed"
            await self.save_state(session_id, state)
            logger.info(f"Session {session_id} fused evidence context updated. Count: {len(fused)}")

    async def add_consensus_step(self, session_id: str, step: ConsensusStep) -> None:
        """Log a complete iteration step of the multi-agent consensus reasoning loop."""
        state = await self.get_state(session_id)
        if state:
            state.history.append(step)
            state.status = f"reasoning_iteration_{step.iteration}"
            await self.save_state(session_id, state)
            logger.info(f"Session {session_id} consensus step {step.iteration} logged. Trust score: {step.trust_score}")

    async def get_all_states(self) -> Dict[str, AgentState]:
        """Retrieve all active session states in memory."""
        async with self._lock:
            return dict(self._memory_db)


# Global Blackboard Instance
blackboard = Blackboard()
