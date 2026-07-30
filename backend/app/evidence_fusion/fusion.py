import numpy as np
import logging
from typing import List
from shared.schema import Evidence
from backend.app.config import settings
from backend.app.database.connections import vector_mgr

logger = logging.getLogger("EvidenceFusion")

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for a 1D array of scores."""
    if len(x) == 0:
        return x
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class EvidenceFuser:
    @staticmethod
    async def fuse(query: str, synchronized_evidence: List[Evidence], top_k: int = 5) -> List[Evidence]:
        """
        Calculates fusion scores using:
        w_i = alpha * relevance_i + beta * confidence_i + gamma * diversity_i + delta * structural_i
        Then applies softmax normalization and picks top-k.
        """
        if not synchronized_evidence:
            return []

        logger.info(f"Fusing {len(synchronized_evidence)} synchronized evidence items...")
        
        # 1. Fetch embeddings for query and all evidence text to calculate semantic similarities
        texts = [ev.content for ev in synchronized_evidence]
        
        # Concurrently fetch embeddings
        embeddings = await vector_mgr.get_embeddings([query] + texts)
        query_emb = embeddings[0]
        ev_embs = embeddings[1:]

        # 2. Compute Semantic Relevance (cosine similarity)
        relevance_scores = []
        for emb in ev_embs:
            # Dot product of normalized vectors
            sim = float(np.dot(query_emb, emb))
            relevance_scores.append(max(0.0, sim))

        # 3. Compute Evidence Diversity
        # Diversity = 1.0 - max similarity to other evidence items
        diversity_scores = []
        n_ev = len(ev_embs)
        for i in range(n_ev):
            max_sim = 0.0
            for j in range(n_ev):
                if i == j:
                    continue
                sim = float(np.dot(ev_embs[i], ev_embs[j]))
                if sim > max_sim:
                    max_sim = sim
            diversity_scores.append(1.0 - max_sim)

        # 4. Compute Structural Consistency
        # Checks if entities or relationships connect structurally to our Knowledge Graph
        structural_scores = []
        for ev in synchronized_evidence:
            if ev.source_type == "graph":
                structural_scores.append(1.0)
            elif ev.source_type == "relational":
                # Relational has high schema integrity
                structural_scores.append(0.8)
            elif ev.source_type == "web":
                # Web evidence: boost if from a trusted authority domain, else default 0.40
                is_trusted = False
                if ev.domain:
                    is_trusted = any(d in ev.domain for d in ["github.com", "arxiv.org", "ieee.org", "acm.org", "microsoft.com", "google.com", "openai.com"])
                structural_scores.append(0.75 if is_trusted else 0.40)
            else:
                # Vector has lower structured backing
                # Boost if it overlaps with graph metadata terms
                if "component" in ev.metadata or "rule" in ev.metadata:
                    structural_scores.append(0.7)
                else:
                    structural_scores.append(0.3)

        # 5. Compute Weighted Score for each item
        raw_scores = []
        for i, ev in enumerate(synchronized_evidence):
            w = (
                settings.ALPHA_RELEVANCE * relevance_scores[i] +
                settings.BETA_CONFIDENCE * ev.confidence +
                settings.GAMMA_DIVERSITY * diversity_scores[i] +
                settings.DELTA_STRUCTURAL * structural_scores[i]
            )
            raw_scores.append(w)

        # 6. Apply Softmax Normalization
        softmax_weights = softmax(np.array(raw_scores))

        # 7. Map weights back to evidence and sort
        fused_evidence = []
        for idx, ev in enumerate(synchronized_evidence):
            ev_copy = ev.model_copy()
            ev_copy.score = float(softmax_weights[idx])
            fused_evidence.append(ev_copy)

        # Sort by score descending
        fused_evidence.sort(key=lambda e: e.score, reverse=True)
        
        logger.info(f"Fusing complete. Selected top {min(top_k, len(fused_evidence))} results.")
        return fused_evidence[:top_k]
