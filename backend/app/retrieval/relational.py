from typing import List
import re
from backend.app.retrieval.base import BaseRetriever
from backend.app.database.connections import db_mgr
from shared.schema import Evidence

class RelationalRetriever(BaseRetriever):
    def __init__(self):
        super().__init__(name="RelationalRetriever")

    async def _retrieve_impl(self, query: str, session_id: str, limit: int) -> List[Evidence]:
        # Identify if we are asking about specific agents or system components
        keywords = ["orchestrator", "blackboard", "planner", "generator", "critic", "validator", "trust", "fusion", "retrieval"]
        target_component = None
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', query.lower()):
                # Map to seeded database primary keys
                if kw == "orchestrator": target_component = "AgentOrchestrator"
                elif kw == "blackboard": target_component = "Blackboard"
                elif kw == "planner": target_component = "Planner"
                elif kw == "generator": target_component = "Generator"
                elif kw == "critic": target_component = "Critic"
                elif kw == "validator": target_component = "Validator"
                elif kw == "trust": target_component = "Trust"
                elif kw == "fusion": target_component = "EvidenceFusion"
                elif kw == "retrieval": target_component = "ParallelRetrieval"

        evidence_list = []

        if target_component:
            # Query component info joined with validation rules
            sql = """
            SELECT c.name, c.type, c.description, c.status, r.rule_name, r.constraint_type, r.min_score
            FROM system_components c
            LEFT JOIN validation_rules r ON c.id = r.target_component
            WHERE c.id = $1
            LIMIT $2;
            """
            rows = await db_mgr.execute_query(sql, target_component, limit)
        else:
            # General query: return matching validation rules or components
            sql = """
            SELECT c.name, c.type, c.description, c.status, r.rule_name, r.constraint_type, r.min_score
            FROM system_components c
            LEFT JOIN validation_rules r ON c.id = r.target_component
            LIMIT $1;
            """
            rows = await db_mgr.execute_query(sql, limit)

        for idx, row in enumerate(rows):
            name = row.get("name", "Unknown Component")
            desc = row.get("description", "")
            status = row.get("status", "unknown")
            rule = row.get("rule_name")
            min_score = row.get("min_score")
            
            content = f"Relational DB record: Component '{name}' (Type: {row.get('type')}) is '{status}'. Description: {desc}"
            if rule:
                content += f" Subject to validation rule '{rule}' ({row.get('constraint_type')}) with min consensus threshold {min_score}."

            evidence_list.append(
                Evidence(
                    id=f"relational_{session_id}_{idx}",
                    content=content,
                    source_type="relational",
                    confidence=1.0,
                    metadata={
                        "component_name": name,
                        "status": status,
                        "rule": rule,
                        "min_score": min_score
                    }
                )
            )

        return evidence_list
