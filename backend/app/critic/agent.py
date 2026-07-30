import logging
from backend.app.llm import call_llm

logger = logging.getLogger("CriticAgent")

class CriticAgent:
    def __init__(self):
        self.system_prompt = """
        You are the Critic Agent for IMPKR-AGENT.
        Your task is to review the candidate response and compare it against the Fused Evidence Context.
        
        Look for:
        1. Logical flaws, weak arguments, or contradictory reasoning.
        2. Factual claims made in the response that are NOT supported by the evidence context.
        3. Key evidence details that are omitted but would improve the completeness of the response.
        
        Provide a concise critique detailing the issues and suggest explicit edits.
        """

    async def critique(self, query: str, response_draft: str, fused_context: str) -> str:
        logger.info("Critiquing candidate response draft...")
        user_prompt = (
            f"Query: {query}\n\n"
            f"Candidate Response Draft:\n{response_draft}\n\n"
            f"Fused Evidence Context:\n{fused_context}\n\n"
            f"Provide your analysis and constructive critique."
        )
        return await call_llm(self.system_prompt, user_prompt)
