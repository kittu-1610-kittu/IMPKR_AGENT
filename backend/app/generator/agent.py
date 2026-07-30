import logging
from backend.app.llm import call_llm

logger = logging.getLogger("GeneratorAgent")

class GeneratorAgent:
    def __init__(self):
        self.system_prompt = """
        You are the Generator Agent for IMPKR-AGENT.
        Your task is to generate a comprehensive, direct, and factual response to the user query.
        
        CRITICAL RULES:
        1. Base your answer STRICTLY on the Fused Evidence Context provided. Do NOT make up information or hallucinate.
        2. Provide inline citations citing the retriever name in brackets, e.g. [VectorRetriever], [GraphRetriever], [RelationalRetriever], [WebRetriever].
        3. If there is a previous draft response and a Critic Critique, address all issues, correct logical flaws, and output an improved draft.
        """

    async def generate_response(self, query: str, fused_context: str, previous_draft: str = "", critique: str = "") -> str:
        logger.info("Generating candidate response...")
        user_prompt = f"Query: {query}\n\nFused Evidence Context:\n{fused_context}\n\n"
        if previous_draft:
            user_prompt += f"Previous Draft Response:\n{previous_draft}\n\n"
            user_prompt += f"Critic Critique:\n{critique}\n\n"
            user_prompt += "Please revise the draft response addressing the critique and maintaining strict source grounding."
        else:
            user_prompt += "Please generate the initial candidate response with inline citations."

        return await call_llm(self.system_prompt, user_prompt)
