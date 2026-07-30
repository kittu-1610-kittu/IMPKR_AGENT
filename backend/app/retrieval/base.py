import time
import logging
from abc import ABC, abstractmethod
from typing import List
from shared.schema import Evidence

logger = logging.getLogger("BaseRetriever")

class BaseRetriever(ABC):
    def __init__(self, name: str):
        self.name = name

    async def retrieve(self, query: str, session_id: str, limit: int = 5) -> List[Evidence]:
        start_time = time.time()
        logger.info(f"[{self.name}] starting retrieval for session {session_id}...")
        try:
            results = await self._retrieve_impl(query, session_id, limit)
            latency = (time.time() - start_time) * 1000  # in ms
            logger.info(f"[{self.name}] completed. Found {len(results)} items in {latency:.2f}ms")
            
            # Inject source type and confidence metadata if missing
            for item in results:
                item.metadata["latency_ms"] = latency
                item.metadata["retriever_name"] = self.name
                
            return results
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"[{self.name}] error during retrieval after {latency:.2f}ms: {e}", exc_info=True)
            return []

    @abstractmethod
    async def _retrieve_impl(self, query: str, session_id: str, limit: int) -> List[Evidence]:
        pass
