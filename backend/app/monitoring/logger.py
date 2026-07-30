import os
import time
import json
import logging
from typing import Dict, Any, List
from pydantic import BaseModel

# Create logs directory at project root
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Helper to write to specific log files
def write_log(filename: str, session_id: str, event_type: str, data: Dict[str, Any]):
    filepath = os.path.join(LOGS_DIR, filename)
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "epoch": time.time(),
        "session_id": session_id,
        "event_type": event_type,
        "payload": data
    }
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


class MetricEntry(BaseModel):
    timestamp: float
    session_id: str
    stage: str
    latency_ms: float
    details: Dict[str, Any]


class TelemetrySystem:
    def __init__(self):
        self.metrics: List[MetricEntry] = []
        self._aggregate_stats = {
            "total_queries_processed": 0,
            "average_latency_ms": 0.0,
            "failed_queries": 0,
            "total_consensus_iterations": 0,
            "validation_rates": []
        }

    def log_stage_latency(self, session_id: str, stage: str, latency_ms: float, details: Dict[str, Any] = None):
        """Record latency metric, update stats, and route to specific log files."""
        entry = MetricEntry(
            timestamp=time.time(),
            session_id=session_id,
            stage=stage,
            latency_ms=latency_ms,
            details=details or {}
        )
        self.metrics.append(entry)
        
        # Keep metrics buffer capped
        if len(self.metrics) > 5000:
            self.metrics.pop(0)

        # Update aggregation stats
        if stage == "orchestration_total":
            self._aggregate_stats["total_queries_processed"] += 1
            total_processed = self._aggregate_stats["total_queries_processed"]
            prev_avg = self._aggregate_stats["average_latency_ms"]
            self._aggregate_stats["average_latency_ms"] = (
                (prev_avg * (total_processed - 1) + latency_ms) / total_processed
            )
            write_log("system.log", session_id, "orchestration_complete", {"latency_ms": latency_ms, "details": details})
        
        elif stage == "planning":
            write_log("reasoning.log", session_id, "planner_complete", {"latency_ms": latency_ms, "plan": details})
            
        elif stage == "retrieval":
            write_log("retrieval.log", session_id, "retrieval_complete", {"latency_ms": latency_ms, "metrics": details})
            
        elif stage == "generator":
            write_log("reasoning.log", session_id, "generator_complete", {"latency_ms": latency_ms, "draft": details})
            
        elif stage == "critic":
            write_log("reasoning.log", session_id, "critic_complete", {"latency_ms": latency_ms, "critique": details})
            
        elif stage == "validator":
            write_log("validation.log", session_id, "validation_complete", {"latency_ms": latency_ms, "results": details})
            
        elif stage == "trust":
            write_log("trust.log", session_id, "trust_complete", {"latency_ms": latency_ms, "assessment": details})

        # Standard structured stdout logging
        log_payload = {
            "telemetry_type": "metric",
            "session_id": session_id,
            "stage": stage,
            "latency_ms": latency_ms,
            "details": details or {}
        }
        # Direct writing to system log as fallback
        write_log("system.log", session_id, f"stage_{stage}", log_payload)

    def log_feedback(self, session_id: str, rating: int, corrections: str, accepted: bool):
        """Log user RLHF feedback data."""
        feedback_payload = {
            "rating": rating,
            "corrections": corrections,
            "accepted": accepted
        }
        write_log("feedback.log", session_id, "user_feedback_submission", feedback_payload)

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Compile overall tracking metrics."""
        val_rates = self._aggregate_stats["validation_rates"]
        avg_val_rate = sum(val_rates) / len(val_rates) if val_rates else 0.90
        
        return {
            "total_queries": self._aggregate_stats["total_queries_processed"],
            "avg_latency_ms": self._aggregate_stats["average_latency_ms"],
            "total_iterations": self._aggregate_stats["total_consensus_iterations"],
            "average_validation_success_rate": avg_val_rate,
            "stage_breakdown_avg_ms": self._calculate_stage_averages()
        }

    def _calculate_stage_averages(self) -> Dict[str, float]:
        stages = ["planning", "retrieval", "fusion", "generator", "critic", "validator", "trust"]
        averages = {}
        for s in stages:
            stage_metrics = [m.latency_ms for m in self.metrics if m.stage == s]
            averages[s] = sum(stage_metrics) / len(stage_metrics) if stage_metrics else 0.0
        return averages

# Global Telemetry & Logging Instance
telemetry = TelemetrySystem()
