import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "IMPKR-AGENT Backend"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # API Keys
    GEMINI_API_KEY: str = Field(default="", env="GEMINI_API_KEY")
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    TAVILY_API_KEY: str = Field(default="", env="TAVILY_API_KEY")
    SERPER_API_KEY: str = Field(default="", env="SERPER_API_KEY")
    BRAVE_API_KEY: str = Field(default="", env="BRAVE_API_KEY")
    BING_API_KEY: str = Field(default="", env="BING_API_KEY")

    # Web Retrieval Integration Configuration
    ENABLE_WEB_RETRIEVAL: bool = Field(default=True, env="ENABLE_WEB_RETRIEVAL")
    WEB_TOP_K: int = Field(default=10, env="WEB_TOP_K")
    WEB_TIMEOUT: float = Field(default=15.0, env="WEB_TIMEOUT")
    WEB_CACHE_TTL: int = Field(default=3600, env="WEB_CACHE_TTL")
    WEB_PROVIDER: str = Field(default="tavily", env="WEB_PROVIDER")

    # Database Settings
    POSTGRES_URL: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5432/impkr_agent", env="POSTGRES_URL")
    NEO4J_URI: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    NEO4J_USER: str = Field(default="neo4j", env="NEO4J_USER")
    NEO4J_PASSWORD: str = Field(default="password", env="NEO4J_PASSWORD")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # Local Mocks & Fallbacks (Highly useful if services aren't running)
    USE_MOCK_LLM: bool = Field(default=False, env="USE_MOCK_LLM")
    USE_MOCK_DATABASES: bool = Field(default=True, env="USE_MOCK_DATABASES")

    # Table 9 Embedding & LLM Specifications
    EMBEDDING_MODEL: str = "microsoft/graphrag/debert-base"
    DEFAULT_MODEL: str = Field(default="CodeLlama-13B-Instruct", env="DEFAULT_MODEL")
    TOP_K: int = 10
    
    # Chunk Specifications
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Graph Traversal Parameters
    MAX_TRAVERSAL_DEPTH: int = 4
    MAX_NEIGHBOR_EXPANSION: int = 12
    EDGE_WEIGHT_CONNECTIVITY: float = 0.45
    ENTITY_MATCH_CONFIDENCE: float = 0.70

    # Similarity thresholds
    SIMILARITY_THRESHOLD: float = 0.78

    # Table 9 Evidence Fusion Coefficients
    ALPHA_RELEVANCE: float = 0.30       # Weight for semantic relevance
    BETA_CONFIDENCE: float = 0.25       # Weight for retrieval confidence
    GAMMA_DIVERSITY: float = 0.20       # Weight for evidence diversity
    DELTA_STRUCTURAL: float = 0.25      # Weight for structural consistency

    # Table 9 Trust & Validation Component Weights [0.35, 0.25, 0.20, 0.20]
    WEIGHT_VALIDATION: float = 0.35     # Verified claim rate weight
    WEIGHT_CONFIDENCE: float = 0.25     # Claim validation confidence weight
    WEIGHT_CONSISTENCY: float = 0.20     # Semantic consistency weight
    WEIGHT_GRAPH_SUPPORT: float = 0.20   # Graph support weight

    # Table 9 Thresholds
    VALIDATION_THRESHOLD: float = 0.75              # \tau_v
    GRAPH_SUPPORT_THRESHOLD: float = 0.70            # \tau_g
    SEMANTIC_CONSISTENCY_THRESHOLD: float = 0.75     # \tau_c
    TRUST_ACCEPTANCE_THRESHOLD: float = 0.85         # \tau_t
    CONVERGENCE_THRESHOLD: float = 0.85              # Alias for trust loop convergence

    # Table 9 Loop Limits
    MAX_AGENT_ITERATIONS: int = 5
    MAX_QUERY_EXPANSION_ITERATIONS: int = 3
    CONSENSUS_CONVERGENCE: float = 1e-3              # \epsilon

    # Table 9 LLM Generation Parameters
    TEMPERATURE: float = 0.2
    TOP_P: float = 0.95
    TOP_K_SAMPLING: int = 40
    MAX_TOKENS: int = 2048
    CONTEXT_WINDOW: int = 4096

    # Random Seed and Evaluation Seeds
    RANDOM_SEED: int = 42
    EVALUATION_SEEDS: list = [42, 52, 62, 72, 82]

    # RLHF Edge Weight Learning Rate
    RLHF_LEARNING_RATE: float = 0.05    # \eta

    # PPO Calibration Settings
    PPO_UPDATE_FREQUENCY: int = 100
    PPO_CLIP: float = 0.2
    LEARNING_RATE: float = 1e-5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
