import json
import logging
from typing import List, Dict, Any
from backend.app.llm import call_llm
from backend.app.config import settings
from shared.schema import ValidationResult

logger = logging.getLogger("TrustAgent")

class TrustAgent:
    def __init__(self):
        self.system_prompt = """
        You are the Trust Agent for IMPKR-AGENT.
        Your task is to review the validation results, intermediate scores, and check if the consensus loop has converged.
        
        You will receive:
        1. Numerical statistics (Verification Rate, Average Claim Confidence, Web Validation Rate, Web Confidence).
        2. Factual claim details and their validations.
        3. Current iteration count and history status.
        
        Evaluate whether the candidate response is factually reliable and meets the convergence criteria.
        Factor in Web validation rate, Web confidence, Web source reliability, and Domain authority of whitelisted/trusted references.
        Convergence threshold is: {threshold}
        
        You MUST respond ONLY with a JSON object:
        {{
            "confidence": 0.0 to 1.0,
            "consensus_score": 0.0 to 1.0,
            "converged": true | false,
            "reliability_assessment": "Detailed summary of response factual trust level."
        }}
        """

    async def assess_trust(self, validation_results: List[ValidationResult], iteration: int) -> Dict[str, Any]:
        logger.info(f"Assessing trust for iteration {iteration}...")
        
        if not validation_results:
            return {
                "confidence": 0.0,
                "consensus_score": 0.0,
                "converged": False,
                "reliability_assessment": "No claims validated yet."
            }

        # Calculate math scores using Table 9 trust weights: 0.35, 0.25, 0.20, 0.20
        total_claims = len(validation_results)
        verified_claims = sum(1 for r in validation_results if r.status == "verified")
        refuted_claims = sum(1 for r in validation_results if r.status == "refuted")
        
        avg_confidence = sum(r.confidence_score for r in validation_results) / total_claims
        verification_rate = verified_claims / total_claims
        
        # Semantic consistency: verified by text documents (vector/web)
        text_verified = sum(1 for r in validation_results if r.status == "verified" and r.evidence_id and ("vector" in r.evidence_id or "web" in r.evidence_id))
        semantic_consistency = text_verified / total_claims
        
        # Graph support: verified by graph paths
        graph_verified = sum(1 for r in validation_results if r.status == "verified" and r.evidence_id and "graph" in r.evidence_id)
        graph_support = graph_verified / total_claims
        
        # Web validation rates & confidence metrics
        web_claims = [r for r in validation_results if r.evidence_id and "web" in r.evidence_id]
        web_validation_rate = sum(1 for r in web_claims if r.status == "verified") / len(web_claims) if web_claims else 1.0
        web_avg_confidence = sum(r.confidence_score for r in web_claims) / len(web_claims) if web_claims else 1.0
        
        # Weighted trust score (0.35, 0.25, 0.20, 0.20)
        trust_score = (
            settings.WEIGHT_VALIDATION * verification_rate +
            settings.WEIGHT_CONFIDENCE * avg_confidence +
            settings.WEIGHT_CONSISTENCY * semantic_consistency +
            settings.WEIGHT_GRAPH_SUPPORT * graph_support
        )
        if refuted_claims > 0:
            # Penalize for refuted claims
            trust_score *= 0.5

        # Format input for LLM assessment
        val_str = ""
        for r in validation_results:
            val_str += f"- Claim: '{r.claim}' | Status: {r.status} | Confidence: {r.confidence_score} | Source: {r.evidence_id}\n"
            
        stats_info = (
            f"Iteration: {iteration}\n"
            f"Total Claims: {total_claims}\n"
            f"Verified Claims: {verified_claims}\n"
            f"Refuted Claims: {refuted_claims}\n"
            f"Verification Rate: {verification_rate:.2f}\n"
            f"Average Confidence: {avg_confidence:.2f}\n"
            f"Calculated Trust Score: {trust_score:.2f}\n"
            f"Web Validation Rate: {web_validation_rate:.2f}\n"
            f"Web Average Confidence: {web_avg_confidence:.2f}\n"
        )
        
        user_prompt = (
            f"Validation Results:\n{val_str}\n\n"
            f"Statistical Summary:\n{stats_info}\n\n"
            f"Determine if the current draft response is convergent."
        )

        sys_prompt = self.system_prompt.format(threshold=settings.CONVERGENCE_THRESHOLD)
        response_str = await call_llm(sys_prompt, user_prompt, json_mode=True)
        
        try:
            data = json.loads(response_str)
            # Ensure types
            confidence = float(data.get("confidence", avg_confidence))
            consensus = float(data.get("consensus_score", verification_rate))
            
            converged = bool(data.get("converged", False))
            
            # If trust score matches or iteration limit reached, force True
            if confidence * consensus >= settings.CONVERGENCE_THRESHOLD:
                converged = True
            if iteration >= settings.MAX_AGENT_ITERATIONS:
                converged = True
                
            return {
                "confidence": confidence,
                "consensus_score": consensus,
                "converged": converged,
                "reliability_assessment": data.get("reliability_assessment", "Assessment processed.")
            }
        except Exception as e:
            logger.error(f"Failed to parse Trust Agent response: {e}. Raw response: {response_str}")
            # Numeric Fallback
            converged = (trust_score >= settings.CONVERGENCE_THRESHOLD) or (iteration >= settings.MAX_AGENT_ITERATIONS)
            return {
                "confidence": avg_confidence,
                "consensus_score": verification_rate,
                "converged": converged,
                "reliability_assessment": f"Numeric fallback assessment. Score: {trust_score:.2f}."
            }
