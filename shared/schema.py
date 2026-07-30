from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Evidence(BaseModel):
    id: str
    content: str
    source_type: str  # "vector" | "graph" | "relational" | "web"
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0  # Computed during fusion

    # Web specific schema fields
    title: Optional[str] = None
    url: Optional[str] = None
    domain: Optional[str] = None
    snippet: Optional[str] = None
    retrieval_time: Optional[float] = None
    embedding: Optional[List[float]] = None
    timestamp: Optional[float] = None

class SubTask(BaseModel):
    id: str
    description: str
    sources: List[str]  # e.g., ["vector", "graph", "relational", "web"]
    status: str = "pending"  # "pending" | "running" | "completed" | "failed"
    result: Optional[str] = None

class PlannerExecutionPlan(BaseModel):
    query: str
    subtasks: List[SubTask]
    rationale: str

class ValidationResult(BaseModel):
    claim: str
    status: str  # "verified" | "refuted" | "uncertain"
    evidence_id: Optional[str] = None
    reasoning: str
    confidence_score: float

class ConsensusStep(BaseModel):
    iteration: int
    candidate_response: str
    critique: str
    validation_results: List[ValidationResult]
    trust_score: float

class FinalResponse(BaseModel):
    query: str
    answer: str
    reasoning_summary: str
    supporting_evidence: List[Evidence]
    validation_score: float
    trust_score: float
    iterations_count: int
    sources: List[Dict[str, Any]]

class AgentState(BaseModel):
    session_id: str
    query: str
    plan: Optional[PlannerExecutionPlan] = None
    raw_evidence: List[Evidence] = Field(default_factory=list)
    fused_evidence: List[Evidence] = Field(default_factory=list)
    history: List[ConsensusStep] = Field(default_factory=list)
    status: str = "initialized"  # "initialized" | "planning" | "retrieving" | "fusing" | "reasoning" | "done"

    # Blackboard integration web fields
    web_results: Optional[List[Dict[str, Any]]] = None
    web_confidence: Optional[float] = None
    web_latency: Optional[float] = None
    web_sources: Optional[List[str]] = None
