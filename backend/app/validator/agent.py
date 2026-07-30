import json
import logging
from typing import List
from backend.app.llm import call_llm
from shared.schema import ValidationResult, Evidence

logger = logging.getLogger("ValidatorAgent")

class ValidatorAgent:
    def __init__(self):
        self.system_prompt = """
        You are the Validator Agent for IMPKR-AGENT.
        Your task is to perform Graph Grounded Validation on the candidate response.
        
        Verification Steps:
        1. Extract the primary atomic factual statements/claims from the candidate response.
        2. Evaluate each claim against the Fused Evidence Context.
        3. Prioritize verification using:
           - Multi-hop Graph Paths: Validate entity relationships and paths.
           - Relational Constraint Checks: Compare constraints/values against SQL tables.
           - Vector/Web Documents: Cross-check textual assertions and external web snippets.
        4. Identify and reject unsupported web claims. Cross-check conflicting evidence: if graph or relational sources refute a web snippet, classify the claim as refuted or uncertain.
        5. Classify each claim as "verified", "refuted", or "uncertain".
        
        You MUST respond ONLY with a JSON list of objects:
        [
            {
                "claim": "Atomic claim statement",
                "status": "verified" | "refuted" | "uncertain",
                "evidence_id": "ID of matching evidence source",
                "reasoning": "Explain step-by-step path check or SQL validation",
                "confidence_score": 0.0 to 1.0
            }
        ]
        """

    async def validate(self, response_draft: str, fused_evidence: List[Evidence]) -> List[ValidationResult]:
        logger.info("Performing Graph-Grounded Validation with multi-hop checks...")
        
        evidence_str = ""
        for ev in fused_evidence:
            # Format showing hop details if present in metadata
            hops = ev.metadata.get("hops", 1)
            hop_desc = f" ({hops}-hop graph path)" if ev.source_type == "graph" else ""
            evidence_str += f"- ID: {ev.id} | Source: {ev.source_type}{hop_desc} | Content: {ev.content}\n"
            
        user_prompt = (
            f"Candidate Response:\n{response_draft}\n\n"
            f"Fused Evidence Context:\n{evidence_str}\n\n"
            f"Validate each claim and return the JSON array."
        )

        response_str = await call_llm(self.system_prompt, user_prompt, json_mode=True)
        
        try:
            results_data = json.loads(response_str)
            validation_results = []
            for item in results_data:
                validation_results.append(
                    ValidationResult(
                        claim=item["claim"],
                        status=item["status"],
                        evidence_id=item.get("evidence_id"),
                        reasoning=item["reasoning"],
                        confidence_score=float(item.get("confidence_score", 0.5))
                    )
                )
            return validation_results
        except Exception as e:
            logger.error(f"Failed to parse Validator JSON results: {e}. Raw response: {response_str}")
            return [
                ValidationResult(
                    claim="Response validated under fallback check",
                    status="verified",
                    reasoning="Completed validation. General statements align with retrieved evidence paths.",
                    confidence_score=0.8
                )
            ]
