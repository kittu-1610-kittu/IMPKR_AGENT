from typing import List
from backend.app.retrieval.base import BaseRetriever
from backend.app.database.connections import vector_mgr
from shared.schema import Evidence

class VectorRetriever(BaseRetriever):
    def __init__(self):
        super().__init__(name="VectorRetriever")

    async def _retrieve_impl(self, query: str, session_id: str, limit: int) -> List[Evidence]:
        raw_results = await vector_mgr.search(query, top_k=limit)
        
        evidence_list = []
        for idx, doc in enumerate(raw_results):
            evidence_list.append(
                Evidence(
                    id=f"vector_{session_id}_{doc.get('id', idx)}",
                    content=doc["content"],
                    source_type="vector",
                    confidence=doc.get("similarity", 0.8),
                    metadata=doc.get("metadata", {})
                )
            )
        return evidence_list
