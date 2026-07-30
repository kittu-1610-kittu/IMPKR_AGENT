# IMPKR-AGENT: Intelligent Multi-Agent Orchestration with Parallel Knowledge Retrieval

IMPKR-AGENT is a backend-only implementation of the multi-agent RAG reasoning architecture outlined in the system research paper. It coordinates a team of specialized agents (Planner, Generator, Critic, Validator, Trust) over a thread-safe Shared Blackboard. 

Heterogeneous evidence is retrieved concurrently from Vector DB, Neo4j, PostgreSQL, and Web search, optimizing query speeds before triggering a convergence consensus reasoning loop.

---

## 📂 Core Folder Layout

```
IMPKR - AGENT/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI entrypoint (REST, SSE, WS, Auth, Limiters)
│   │   ├── config.py                   # Central configurations & fusion coefficients
│   │   ├── llm.py                      # Circuit Breakers & Exponential Retry handlers
│   │   ├── orchestrator/               # Workflow orchestrator service
│   │   ├── planner/                    # Query decomposition Planner Agent
│   │   ├── retrieval/                  # Parallel retrievers (Vector, Graph, SQL, Web)
│   │   ├── evidence_fusion/            # Schema normalization and Softmax fuser
│   │   ├── blackboard/                 # Centralized shared memory blackboard
│   │   ├── generator/                  # Grounded draft generator
│   │   ├── critic/                     # Draft evaluator
│   │   ├── validator/                  # Graph-grounded validator
│   │   ├── trust/                      # Trust scoring and convergence checking
│   │   ├── monitoring/                 # Multi-file logger & telemetry metrics
│   │   └── database/                   # DB connection adapters and seeder script
│   └── tests/                          # Integration, timing, and security tests
├── shared/
│   └── schema.py                       # Core transport schemas (Evidence, SubTask, etc.)
├── deployment/
│   ├── Dockerfile.backend              # Docker recipe for Python server
│   └── docker-compose.yml              # DB stack orchestration (Postgres, Neo4j, Redis)
├── logs/                               # Structured stage logs (retrieval.log, reasoning.log, etc.)
├── docs/                               # Setup & Architecture Markdown guides
├── cli.py                              # Interactive VS Code Terminal CLI Client
└── run.py                              # Entrypoint wrapper
```

---

## 🧬 Architectural Flow

```
User Query
    ↓
Planner Agent (Subtask Decompositor)
    ↓
Parallel Retriever (Vector DB, Neo4j Graph DB, PostgreSQL SQL, Web Search) [T = max(T_i)]
    ↓
Evidence Synchronizer (Deduplication)
    ↓
Adaptive Evidence Fusion (Softmax relevance, diversity, confidence, KG consistency)
    ↓
Shared Blackboard Memory (State holder)
    ↓
Consensus Loop (Generator ⇄ Critic ⇄ Validator ⇄ Trust Agent)
    ↓
Final Citations Response
    ↓
RLHF Feedback Loop (KG updates & calibration updates)
```

---

## 📖 Complete Documentation

Please refer to the following documents in the `docs/` folder for installation, execution, architecture, and testing instructions:
1. [INSTALL.md](file:///e:/hari%20main%20project/IMPKR%20-%20AGENT/docs/INSTALL.md): Package installation & DB setups.
2. [RUN.md](file:///e:/hari%20main%20project/IMPKR%20-%20AGENT/docs/RUN.md): Launching FastAPI and starting the CLI.
3. [API.md](file:///e:/hari%20main%20project/IMPKR%20-%20AGENT/docs/API.md): Endpoint routes, Auth, and Rate Limits.
4. [ARCHITECTURE.md](file:///e:/hari%20main%20project/IMPKR%20-%20AGENT/docs/ARCHITECTURE.md): Multi-Agent & Fusion details.
5. [CLI.md](file:///e:/hari%20main%20project/IMPKR%20-%20AGENT/docs/CLI.md): Interactive terminal instructions.
6. [TESTING.md](file:///e:/hari%20main%20project/IMPKR%20-%20AGENT/docs/TESTING.md): Running pytest timing checks.

---

## 🌐 Web Retrieval Agent Integration
The Web Retrieval Agent executes in parallel with Vector, Graph, and SQL databases. It features:
- **Search Providers**: Tavily, Serper, Brave, Bing, and DuckDuckGo (instant summaries/related topics).
- **Domain Whitelisting**: Restricts search results to trusted domains (GitHub, ArXiv, Python, Neo4j, Microsoft Learn, AWS, GCP, etc.) and filters out spam, ads, or clickbait domains.
- **Embedded Similarity Ranking**: Generates L2-normalized embeddings via FAISS to compute semantic cosine similarity scores against optimized queries.
- **Blackboard Logging**: Telemetry metrics (`web_results`, `web_confidence`, `web_latency`, and `web_sources`) are tracked on the Blackboard state.
- **Circuit Breakers & Caching**: Independent circuit breakers guard each provider. High-speed query caching is backed by Redis (or in-memory cache fallbacks).

### Configuration Environmental Variables
Add these values to your `.env` file to customize retrieval:
```env
ENABLE_WEB_RETRIEVAL=True
WEB_PROVIDER=tavily
TAVILY_API_KEY=your_key
SERPER_API_KEY=your_key
BRAVE_API_KEY=your_key
BING_API_KEY=your_key
WEB_TIMEOUT=15.0
WEB_CACHE_TTL=3600
WEB_TOP_K=10
```
