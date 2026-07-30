# Architecture & Multi-Agent Design

IMPKR-AGENT is built around a centralized thread-safe blackboard that prevents ad-hoc, peer-to-peer agent messaging. Agents coordinate by writing and reading structured session states.

---

## 1. Multi-Agent Pipeline Breakdown

### A. Planner Agent
- Decomposes the user query into parallel subtasks and selects relevant search adapters (Vector, Graph, Relational, Web).

### B. Parallel Retrievers
- Concurrently queries Vector FAISS, Neo4j, PostgreSQL, and Web search APIs in parallel threads to achieve latency matching $\max(T_i)$.

### C. Evidence Sync & Fusion Engine
- **Synchronizer**: Removes duplicates and overlaps using a Jaccard token coefficient match.
- **Fuser**: Computes normalized Softmax weights to combine documents according to query relevance, source confidence, and evolutionary feedback.

### D. Shared Blackboard
- Thread-safe repository managing active execution states, reasoning histories, plan revisions, and final outputs.

### E. Generator Agent
- Generates a citations-anchored response using bracketed annotations matching evidence items.

### F. Critic Agent
- Evaluates generator drafts against fused context to highlight missing details or potential hallucinations.

### G. Validator Agent
- Extracts claims from candidate response drafts and performs multi-hop entity relation checking on the Neo4j Knowledge Graph.

### H. Trust Agent
- Aggregates validator reports and critic suggestions, scoring trust convergence. Triggers additional revisions if scores fall below thresholds.

---

## 2. Adaptive Calibration & RLHF Updates

Fusing evidence uses weights calculated dynamically from source confidence and relevance scores:

$$S_i = \text{softmax}(w_i)$$

When feedback is received (via rating and corrections), the feedback system:
1. Adjusts source weights (e.g., boosting a source if its rating is high).
2. Updates Neo4j properties or inserts new fact relations based on corrections.

---

## 3. Web Retrieval Integration & Parallel Latency
The Web Retrieval Agent runs in parallel with other databases:

```
                  ┌──────────────────────┐
                  │      User Query      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    Planner Agent     │
                  └──────────┬───────────┘
                             │
            ┌────────────────┼────────────────┬──────────────┐
            ▼                ▼                ▼              ▼
     ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ ┌────────────┐
     │  Vector DB   │ │  Knowledge   │ │ Relational  │ │    Web     │
     │  Retriever   │ │   Graph DB   │ │  Database   │ │ Retriever  │
     └──────┬───────┘ └──────┬───────┘ └──────┬──────┘ └─────┬──────┘
            │                │                │              │
            └────────────────┼────────────────┼──────────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Evidence Sync & Fuse │
                  └──────────────────────┘
```

The retrieval time latency bounds are governed by the max of all active retrievers:
\[T_{\text{parallel}} = \max(T_{\text{vector}}, T_{\text{graph}}, T_{\text{relational}}, T_{\text{web}})\]

### Trust & Validation Formula Integration
The overall validator and trust agent incorporate web evidence by computing:
- **Web Validation Rate**: The proportion of web-grounded claims verified successfully.
- **Web Average Confidence**: The mean confidence score of verified web sources.
- **Domain Authority Score**: A bonus score weighting trusted whitelisted domains.
- **Trust Index**:
\[\text{Trust Score} = 0.35 \times \text{ValidationRate} + 0.25 \times \text{AvgConfidence} + 0.20 \times \text{SemanticConsistency} + 0.20 \times \text{GraphSupport}\]
