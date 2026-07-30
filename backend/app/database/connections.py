import logging
import json
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from backend.app.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DatabaseConnections")

# --- FAISS & Embeddings Fallback ---
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS is available.")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS is not installed. Using numpy-based pure-Python vector search fallback.")

# --- Neo4j Fallback ---
try:
    from neo4j import AsyncGraphDriver, AsyncSession
    NEO4J_AVAILABLE = True
    logger.info("Neo4j driver is available.")
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("Neo4j driver is not installed. Using in-memory graph database fallback.")

# --- asyncpg (PostgreSQL) Fallback ---
try:
    import asyncpg
    POSTGRES_AVAILABLE = True
    logger.info("asyncpg driver is available.")
except ImportError:
    POSTGRES_AVAILABLE = False
    logger.warning("asyncpg is not installed. Using SQLite-based database fallback.")

# --- Redis Fallback ---
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
    logger.info("aioredis driver is available.")
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("aioredis is not installed. Using in-memory Redis fallback.")


# =====================================================================
# REDIS / IN-MEMORY CACHE
# =====================================================================
class InMemoryRedis:
    def __init__(self):
        self._data = {}
        logger.info("Initialized InMemoryRedis Cache fallback.")

    async def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self._data[key] = value
        # Simulating expiration by not actually deleting, or doing it if needed.
        return True

    async def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    async def ping(self) -> bool:
        return True

    async def close(self):
        pass

class RedisConnectionManager:
    def __init__(self):
        self.client = None
        self.is_mock = True

    async def connect(self):
        if settings.USE_MOCK_DATABASES or not REDIS_AVAILABLE:
            self.client = InMemoryRedis()
            self.is_mock = True
            return

        try:
            self.client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            await self.client.ping()
            self.is_mock = False
            logger.info("Connected to Redis successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis ({e}). Falling back to InMemoryRedis.")
            self.client = InMemoryRedis()
            self.is_mock = True

    async def close(self):
        if self.client:
            await self.client.close()


# =====================================================================
# POSTGRESQL / SQLITE DATABASE
# =====================================================================
class RelationalConnectionManager:
    def __init__(self):
        self.pool = None
        self.sqlite_conn = None
        self.is_mock = True

    async def connect(self):
        if settings.USE_MOCK_DATABASES or not POSTGRES_AVAILABLE:
            await self._connect_sqlite()
            return

        try:
            # Parse asyncpg URL (postgres://... or postgresql://...)
            # asyncpg needs postgresql://
            url = settings.POSTGRES_URL
            if url.startswith("postgresql+asyncpg://"):
                url = url.replace("postgresql+asyncpg://", "postgresql://")
            
            self.pool = await asyncpg.create_pool(url, min_size=1, max_size=10)
            self.is_mock = False
            logger.info("Connected to PostgreSQL pool successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL ({e}). Falling back to SQLite.")
            await self._connect_sqlite()

    async def _connect_sqlite(self):
        import sqlite3
        self.sqlite_conn = sqlite3.connect("impkr_agent_fallback.db", check_same_thread=False)
        self.sqlite_conn.row_factory = sqlite3.Row
        self.is_mock = True
        logger.info("Initialized local SQLite database fallback (impkr_agent_fallback.db).")

    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        if not self.is_mock and self.pool:
            async with self.pool.acquire() as connection:
                rows = await connection.fetch(query, *args)
                return [dict(r) for r in rows]
        else:
            # Execute synchronously on SQLite, but run in executor
            loop = asyncio.get_event_loop()
            
            # Map postgres style params ($1, $2) to sqlite style (?)
            sqlite_query = query
            for i in range(10, 0, -1):
                sqlite_query = sqlite_query.replace(f"${i}", "?")
            
            def run():
                cursor = self.sqlite_conn.cursor()
                try:
                    cursor.execute(sqlite_query, args)
                    if cursor.description:
                        columns = [col[0] for col in cursor.description]
                        return [dict(zip(columns, row)) for row in cursor.fetchall()]
                    self.sqlite_conn.commit()
                    return []
                except Exception as ex:
                    logger.error(f"SQLite error: {ex} on query: {sqlite_query}")
                    return []
                finally:
                    cursor.close()

            return await loop.run_in_executor(None, run)

    async def close(self):
        if self.pool:
            await self.pool.close()
            self.pool = None
        if self.sqlite_conn:
            self.sqlite_conn.close()
            self.sqlite_conn = None


# =====================================================================
# NEO4J / IN-MEMORY GRAPH DATABASE
# =====================================================================
class InMemoryGraph:
    def __init__(self):
        self.nodes = {}  # id -> {label: str, properties: dict}
        self.edges = []  # [{source: str, target: str, type: str, confidence: float}]
        logger.info("Initialized InMemoryGraph Database fallback.")

    def add_node(self, node_id: str, label: str, properties: Dict[str, Any]):
        self.nodes[node_id] = {"label": label, "properties": properties}

    def add_edge(self, source: str, target: str, rel_type: str, confidence: float = 1.0):
        self.edges.append({
            "source": source,
            "target": target,
            "type": rel_type,
            "confidence": confidence
        })

    async def query_subgraph(self, entity_name: str, depth: int = 2) -> List[Dict[str, Any]]:
        # Find matching nodes
        start_nodes = []
        for nid, val in self.nodes.items():
            name = val["properties"].get("name", "").lower()
            if entity_name.lower() in name or nid.lower() == entity_name.lower():
                start_nodes.append(nid)

        if not start_nodes:
            return []

        visited = set(start_nodes)
        sub_edges = []
        current_layer = set(start_nodes)

        for _ in range(depth):
            next_layer = set()
            for edge in self.edges:
                src, tgt = edge["source"], edge["target"]
                if src in current_layer or tgt in current_layer:
                    sub_edges.append(edge)
                    if src not in visited:
                        visited.add(src)
                        next_layer.add(src)
                    if tgt not in visited:
                        visited.add(tgt)
                        next_layer.add(tgt)
            current_layer = next_layer

        result = []
        for edge in sub_edges:
            src_node = self.nodes.get(edge["source"])
            tgt_node = self.nodes.get(edge["target"])
            if src_node and tgt_node:
                result.append({
                    "source": {
                        "id": edge["source"],
                        "label": src_node["label"],
                        "properties": src_node["properties"]
                    },
                    "relationship": {
                        "type": edge["type"],
                        "confidence": edge["confidence"]
                    },
                    "target": {
                        "id": edge["target"],
                        "label": tgt_node["label"],
                        "properties": tgt_node["properties"]
                    }
                })
        return result


class GraphConnectionManager:
    def __init__(self):
        self.driver = None
        self.mock_db = None
        self.is_mock = True

    async def connect(self):
        if settings.USE_MOCK_DATABASES or not NEO4J_AVAILABLE:
            self.mock_db = InMemoryGraph()
            self.is_mock = True
            await self._seed_mock_graph()
            return

        try:
            # Neo4j python driver async connection
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI, 
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            self.is_mock = False
            logger.info("Connected to Neo4j successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j ({e}). Falling back to InMemoryGraph.")
            self.mock_db = InMemoryGraph()
            self.is_mock = True
            await self._seed_mock_graph()

    async def _seed_mock_graph(self):
        # Insert some initial sample nodes and edges representing knowledge graph data
        self.mock_db.add_node("AgentOrchestrator", "Architecture", {"name": "Agent Orchestrator", "description": "Coordinates multi-agent flow and blackboard updates."})
        self.mock_db.add_node("Blackboard", "Pattern", {"name": "Shared Blackboard", "description": "Central communication repository for agents."})
        self.mock_db.add_node("Planner", "Agent", {"name": "Planner Agent", "description": "Decomposes queries and routes retrieval."})
        self.mock_db.add_node("Generator", "Agent", {"name": "Generator Agent", "description": "Generates source-grounded responses."})
        self.mock_db.add_node("Critic", "Agent", {"name": "Critic Agent", "description": "Reviews drafts for logical gaps."})
        self.mock_db.add_node("Validator", "Agent", {"name": "Validator Agent", "description": "Factual verification against data sources."})
        self.mock_db.add_node("Trust", "Agent", {"name": "Trust Agent", "description": "Computes consensus metrics and signals convergence."})
        self.mock_db.add_node("ParallelRetrieval", "System", {"name": "Parallel Retrieval", "description": "Simultaneously queries Vector, Graph, PostgreSQL and Web."})
        self.mock_db.add_node("EvidenceFusion", "System", {"name": "Adaptive Evidence Fusion", "description": "Applies Softmax weights to score and fuse retrieved facts."})

        self.mock_db.add_edge("AgentOrchestrator", "Blackboard", "USES", 0.95)
        self.mock_db.add_edge("AgentOrchestrator", "Planner", "INVOKES", 0.90)
        self.mock_db.add_edge("AgentOrchestrator", "ParallelRetrieval", "TRIGGERS", 0.92)
        self.mock_db.add_edge("ParallelRetrieval", "EvidenceFusion", "FEEDS", 0.88)
        self.mock_db.add_edge("EvidenceFusion", "Blackboard", "WRITES_TO", 0.94)
        self.mock_db.add_edge("AgentOrchestrator", "Generator", "INVOKES", 0.90)
        self.mock_db.add_edge("Generator", "Critic", "DRAFT_SENT_TO", 0.85)
        self.mock_db.add_edge("Critic", "Validator", "CRITIQUE_SENT_TO", 0.80)
        self.mock_db.add_edge("Validator", "Trust", "VALIDATION_REPORT_SENT_TO", 0.87)

    async def execute_cypher(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.is_mock and self.driver:
            # Async execution using Neo4j
            def run_session(tx):
                res = tx.run(cypher, **(params or {}))
                return [dict(record) for record in res]

            # In older drivers run is sync, or we run in Executor
            loop = asyncio.get_event_loop()
            with self.driver.session() as session:
                return await loop.run_in_executor(None, session.execute_read, run_session)
        else:
            # Match entities in InMemoryGraph by analyzing cypher queries roughly
            # This is a very lightweight parser for testing queries
            entity_name = ""
            if params and "entity_name" in params:
                entity_name = params["entity_name"]
            elif "name" in cypher.lower():
                # Extract some string
                import re
                match = re.search(r"name\s*[:=]\s*['\"]([^'\"]+)['\"]", cypher, re.IGNORECASE)
                if match:
                    entity_name = match.group(1)
            
            if not entity_name:
                # return all nodes
                res = []
                for nid, val in self.mock_db.nodes.items():
                    res.append({
                        "node": {
                            "id": nid,
                            "label": val["label"],
                            "properties": val["properties"]
                        }
                    })
                return res

            return await self.mock_db.query_subgraph(entity_name)

    async def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None
        self.mock_db = None


# =====================================================================
# VECTOR DB / FAISS / NUMPY MANAGER
# =====================================================================
class VectorDBManager:
    def __init__(self):
        self.index = None
        self.dimension = 768  # Table 9 dimension for microsoft/graphrag/debert-base
        self.documents = []    # Index-aligned documents [{id, content, metadata}]
        self.is_mock = True
        self.model = None

    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings. If not settings.USE_MOCK_DATABASES and SentenceTransformers is available,
        we use the Table 9 embedding model microsoft/graphrag/debert-base.
        Otherwise, we generate L2 normalized mock embeddings based on text hashes.
        """
        if not settings.USE_MOCK_DATABASES:
            try:
                from sentence_transformers import SentenceTransformer
                if self.model is None:
                    # Load Table 9 embedding model
                    self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
                
                # Generate embeddings
                embeddings = self.model.encode(texts, convert_to_numpy=True)
                # Ensure L2 normalization for Cosine Similarity (IndexFlatIP)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                normalized = embeddings / np.where(norms == 0, 1.0, norms)
                return normalized.astype(np.float32)
            except Exception as e:
                logger.error(f"Failed to fetch real Table 9 embeddings ({e}). Falling back to mock embeddings.")

        # Robust Mock Embeddings (consistent L2 normalized vector representation per word/phrase)
        embeddings = []
        for text in texts:
            state = np.random.RandomState(hash(text) % (2**32))
            vector = state.randn(self.dimension)
            # L2 normalize
            norm = np.linalg.norm(vector)
            vector /= (norm if norm != 0 else 1.0)
            embeddings.append(vector)
        
        return np.array(embeddings, dtype=np.float32)

    async def initialize(self):
        self.documents = []
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension) # Cosine similarity (inner product on normalized vectors)
            self.is_mock = False
            logger.info("Initialized FAISS Index.")
        else:
            self.index = None
            self.is_mock = True
            logger.info("Initialized NumPy-based vector store fallback.")

        # Seed sample documents
        await self.add_documents([
            {
                "id": "doc1", 
                "content": "IMPKR-AGENT implements parallel knowledge retrieval where latency is defined as T_parallel = max(T_vector, T_graph, T_relational, T_web). This is superior to sequential pipeline models.",
                "metadata": {"section": "performance"}
            },
            {
                "id": "doc2", 
                "content": "Adaptive Evidence Fusion resolves contradictory information by normalizing relevance, confidence, diversity, and structural consistency scores using a Softmax function.",
                "metadata": {"section": "fusion"}
            },
            {
                "id": "doc3",
                "content": "The Shared Blackboard Architecture serves as a central memory repository where all agents (Planner, Generator, Critic, Validator, Trust) read and write states asynchronously, preventing direct agent-to-agent communication.",
                "metadata": {"section": "architecture"}
            },
            {
                "id": "doc4",
                "content": "Graph Grounded Validation in the Validator Agent extracts atomic statements from candidate answers and performs strict checks against Neo4j relationships and relational schemas.",
                "metadata": {"section": "validation"}
            },
            {
                "id": "doc5",
                "content": "The Trust Agent computes confidence, consensus, and reliability metrics. It compares scores against the Convergence Threshold to check if the Generator's answer requires another iteration.",
                "metadata": {"section": "trust"}
            }
        ])

    async def add_documents(self, docs: List[Dict[str, Any]]):
        texts = [d["content"] for d in docs]
        embeddings = await self.get_embeddings(texts)
        
        for doc in docs:
            self.documents.append(doc)

        if not self.is_mock and self.index:
            self.index.add(embeddings)
        else:
            # NumPy based fallback
            if not hasattr(self, "_numpy_vectors"):
                self._numpy_vectors = []
            
            for emb in embeddings:
                self._numpy_vectors.append(emb)

    async def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.documents:
            return []

        query_emb = await self.get_embeddings([query])
        
        if not self.is_mock and self.index:
            scores, indices = self.index.search(query_emb, min(top_k, len(self.documents)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.documents):
                    continue
                doc = self.documents[idx].copy()
                doc["similarity"] = float(score)
                results.append(doc)
            return results
        else:
            # NumPy manual search
            vectors = np.array(self._numpy_vectors)
            # Dot product (since both are normalized, this is cosine similarity)
            scores = np.dot(vectors, query_emb[0])
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                doc = self.documents[idx].copy()
                doc["similarity"] = float(scores[idx])
                results.append(doc)
            return results


# Global Connection Instances
redis_mgr = RedisConnectionManager()
db_mgr = RelationalConnectionManager()
graph_mgr = GraphConnectionManager()
vector_mgr = VectorDBManager()

async def initialize_all_databases():
    logger.info("Initializing all database connections...")
    await redis_mgr.connect()
    await db_mgr.connect()
    await graph_mgr.connect()
    await vector_mgr.initialize()
    logger.info("Database initializations completed.")

async def close_all_databases():
    logger.info("Closing database connections...")
    await redis_mgr.close()
    await db_mgr.close()
    await graph_mgr.close()
    logger.info("Database connections closed.")
