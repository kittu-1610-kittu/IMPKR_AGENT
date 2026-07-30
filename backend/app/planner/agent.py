import json
import logging
from backend.app.llm import call_llm
from shared.schema import PlannerExecutionPlan, SubTask

logger = logging.getLogger("PlannerAgent")

class PlannerAgent:
    def __init__(self):
        self.system_prompt = """
        You are the Planner Agent for IMPKR-AGENT.
        Your task is to analyze the user query, detect the task type, and decompose it into a set of 2 to 4 parallel subtasks.
        Each subtask must list the data sources it needs to retrieve from.
        Available data sources: ["vector", "graph", "relational", "web"]
        
        You must respond ONLY with a JSON object containing:
        {
            "query": "original user query",
            "subtasks": [
                {
                    "id": "subtask_1",
                    "description": "description of retrieval subtask",
                    "sources": ["vector", "graph"],
                    "status": "pending"
                }
            ],
            "rationale": "short explanation of the plan"
        }
        """

    async def generate_plan(self, query: str) -> PlannerExecutionPlan:
        logger.info(f"Generating execution plan for query: '{query}'")
        user_prompt = f"User Query: {query}"
        
        response_str = await call_llm(self.system_prompt, user_prompt, json_mode=True)
        
        try:
            data = json.loads(response_str)
            subtasks = []
            for t in data.get("subtasks", []):
                subtasks.append(
                    SubTask(
                        id=t["id"],
                        description=t["description"],
                        sources=t["sources"],
                        status="pending"
                    )
                )
            return PlannerExecutionPlan(
                query=query,
                subtasks=subtasks,
                rationale=data.get("rationale", "Decomposed query for parallel search.")
            )
        except Exception as e:
            logger.error(f"Failed to parse Planner JSON response: {e}. Raw response: {response_str}")
            # Robust fallback plan
            return PlannerExecutionPlan(
                query=query,
                subtasks=[
                    SubTask(id="subtask_vector", description="Fetch general vector facts.", sources=["vector"]),
                    SubTask(id="subtask_graph", description="Fetch graph relationships.", sources=["graph"]),
                    SubTask(id="subtask_relational", description="Fetch relational component schemas.", sources=["relational"]),
                    SubTask(id="subtask_web", description="Fetch web articles.", sources=["web"])
                ],
                rationale="Fallback parallel plan generated due to parsing error."
            )
        
