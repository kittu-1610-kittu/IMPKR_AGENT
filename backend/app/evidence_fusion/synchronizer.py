import logging
from typing import List
from shared.schema import Evidence

logger = logging.getLogger("EvidenceSynchronizer")

def token_jaccard_similarity(str1: str, str2: str) -> float:
    """Calculate Jaccard similarity between word tokens of two strings."""
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

class EvidenceSynchronizer:
    @staticmethod
    def synchronize(raw_evidence: List[Evidence]) -> List[Evidence]:
        """
        Merge, normalize, and deduplicate evidence from various heterogeneous sources.
        Resolves conflicts by prioritising sources: graph > relational > vector > web.
        """
        if not raw_evidence:
            return []

        # 1. Schema Normalization & Source Type enforcement
        normalized = []
        for ev in raw_evidence:
            # Ensure confidence is within bounds [0, 1]
            confidence = max(0.0, min(1.0, ev.confidence))
            
            # Source confidence adjustments
            source_weight = 1.0
            if ev.source_type == "graph":
                source_weight = 0.95
            elif ev.source_type == "relational":
                source_weight = 0.90
            elif ev.source_type == "vector":
                source_weight = 0.80
            elif ev.source_type == "web":
                source_weight = 0.70

            normalized.append(
                Evidence(
                    id=ev.id,
                    content=ev.content.strip(),
                    source_type=ev.source_type,
                    confidence=confidence * source_weight,
                    metadata=ev.metadata or {},
                    title=ev.title,
                    url=ev.url,
                    domain=ev.domain,
                    snippet=ev.snippet,
                    retrieval_time=ev.retrieval_time,
                    embedding=ev.embedding,
                    timestamp=ev.timestamp
                )
            )

        # 2. Deduplication using token overlap
        deduplicated: List[Evidence] = []
        similarity_threshold = 0.70  # Mark as duplicates if Jaccard similarity > 70%

        for candidate in normalized:
            is_duplicate = False
            for existing in deduplicated:
                sim = token_jaccard_similarity(candidate.content, existing.content)
                if sim > similarity_threshold:
                    is_duplicate = True
                    # Conflict resolution: Keep the one with higher source/retrieval confidence
                    if candidate.confidence > existing.confidence:
                        # Replace existing with the higher confidence version
                        existing.content = candidate.content
                        existing.confidence = candidate.confidence
                        existing.source_type = candidate.source_type
                        existing.metadata.update(candidate.metadata)
                        # Copy web specific schema fields
                        existing.title = candidate.title
                        existing.url = candidate.url
                        existing.domain = candidate.domain
                        existing.snippet = candidate.snippet
                        existing.retrieval_time = candidate.retrieval_time
                        existing.embedding = candidate.embedding
                        existing.timestamp = candidate.timestamp
                    break
            
            if not is_duplicate:
                deduplicated.append(candidate)

        logger.info(f"Synchronized {len(raw_evidence)} raw items down to {len(deduplicated)} deduplicated items.")
        return deduplicated
