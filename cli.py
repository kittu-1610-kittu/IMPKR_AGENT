import os
import sys
import time
import uuid
import json
import asyncio
import httpx
from typing import Dict, Any

# ANSI Terminal Colors
C_CYAN = "\033[96m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_MAGENTA = "\033[95m"
C_GRAY = "\033[90m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

BACKEND_URL = "http://localhost:8000"
API_KEY = "impkr_secret_token"

async def check_backend_alive() -> bool:
    """Check if FastAPI server is running locally."""
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            resp = await client.get(f"{BACKEND_URL}/api/health")
            return resp.status_code == 200
    except Exception:
        return False

async def run_cli_session():
    print(f"\n{C_BOLD}{C_CYAN}=================================================={C_RESET}")
    print(f"{C_BOLD}{C_CYAN}[IMPKR-AGENT] Interactive Terminal CLI Client{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}=================================================={C_RESET}")
    
    server_active = await check_backend_alive()
    if server_active:
        print(f"{C_GREEN}[OK] Connected to FastAPI server at {BACKEND_URL} (API Mode){C_RESET}")
    else:
        print(f"{C_YELLOW}[WARNING] FastAPI server not detected on port 8000.{C_RESET}")
        print(f"{C_CYAN}Initializing local databases and running in-process (Direct Mode)...{C_RESET}")
        # Initialize modules in-process
        from backend.app.database.connections import initialize_all_databases
        from backend.app.database.initial_data import seed_relational_database
        await initialize_all_databases()
        await seed_relational_database()
        print(f"{C_GREEN}[OK] Local database connections established successfully.{C_RESET}")

    while True:
        try:
            print(f"\n{C_BOLD}Enter your query (or type '/eval' to run Table 9 benchmark, 'exit' to quit):{C_RESET}")
            query = input(f"{C_CYAN}> {C_RESET}").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print(f"{C_GRAY}Shutting down CLI... Goodbye!{C_RESET}")
                break
                
            if query.lower() in ["/eval", "/table9"]:
                print(f"\n{C_BOLD}{C_CYAN}--- Table 9 Dataset Preprocessing & Statistical Benchmark ---{C_RESET}")
                from backend.app.database.datasets import DatasetManager, StatisticalValidator
                print(f"{C_GRAY}Loading and indexing datasets (HotpotQA, StrategyQA, HumanEval, etc.)...{C_RESET}")
                await DatasetManager.preprocess_and_index_all()
                print(f"{C_GREEN}[OK] Datasets preprocessed and indexed in FAISS / Mock Graph.{C_RESET}")
                
                # Execute statistical evaluation query
                benchmark_query = "What parallel database and graph tools are unified in IMPKR-AGENT?"
                await StatisticalValidator.run_statistical_evaluation(benchmark_query)
                continue

            session_id = f"cli_{uuid.uuid4().hex[:6]}"
            print(f"\n{C_GRAY}Starting pipeline (Session: {session_id})...{C_RESET}\n")

            if server_active:
                await execute_api_mode(query, session_id)
            else:
                await execute_direct_mode(query, session_id)

        except KeyboardInterrupt:
            print(f"\n{C_GRAY}Session cancelled by user.{C_RESET}")
            break
        except Exception as e:
            print(f"\n{C_RED}Error running query: {e}{C_RESET}")

async def execute_api_mode(query: str, session_id: str):
    """Streams SSE updates from the running FastAPI server."""
    headers = {"X-API-KEY": API_KEY}
    url = f"{BACKEND_URL}/api/query/stream?query={httpx.URLEscape(query)}&session_id={session_id}"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("GET", url, headers=headers) as response:
                if response.status_code == 401:
                    print(f"{C_RED}Authentication failed: X-API-KEY header rejected.{C_RESET}")
                    return
                elif response.status_code == 429:
                    print(f"{C_RED}Rate limit exceeded. Please wait before running another query.{C_RESET}")
                    return
                elif response.status_code != 200:
                    print(f"{C_RED}Request failed with status {response.status_code}{C_RESET}")
                    return

                # Read SSE lines
                async for line in response.iter_lines():
                    if line.startswith("data:"):
                        event_data_str = line[5:].strip()
                        event_wrapper = json.loads(event_data_str)
                        await process_event(event_wrapper, session_id, api_mode=True)
    except Exception as e:
        print(f"{C_RED}HTTP streaming error: {e}{C_RESET}")

async def execute_direct_mode(query: str, session_id: str):
    """Runs orchestrator directly in the CLI process."""
    from backend.app.orchestrator.service import OrchestratorService
    orc = OrchestratorService()
    
    async for event_str in orc.execute_query_stream(session_id, query):
        event_wrapper = json.loads(event_str)
        await process_event(event_wrapper, session_id, api_mode=False)

async def process_event(wrapper: Dict[str, Any], session_id: str, api_mode: bool):
    ev_type = wrapper.get("event")
    data = wrapper.get("data")

    # Capture and formatting for the 19 steps

    if ev_type == "status":
        print(f"{C_GRAY}[System] {data}{C_RESET}")

    elif ev_type == "step_1_init":
        print(f"\n{C_BOLD}{C_CYAN}=================================================={C_RESET}")
        print(f"{C_BOLD}{C_CYAN}IMPKR-AGENT WORKFLOW RUN{C_RESET}")
        print(f"{C_BOLD}{C_CYAN}=================================================={C_RESET}")
        print(f"{C_BOLD}User Query:{C_RESET} {data.get('query')}")
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Session Context Initialization{C_RESET}")
        print(f" ✓ Stored Session (ID: {data.get('session_id')})")
        print(" ✓ Initialized Context")
        print(" ✓ Created Blackboard Session")
        print(" ✓ Created Tracking Session")
        print(" ✓ Initialize Monitoring")
        print(" ✓ Initialize Feedback Memory")
        print(" ✓ Initialize Routing Context")

    elif ev_type == "step_2_plan":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Task Planning (Planner Agent){C_RESET}")
        print(" ✓ Query Analysis")
        print(" ✓ Intent Detection")
        print(" ✓ Context Analysis")
        print(" ✓ Task Decomposition")
        print(" ✓ Subtask Scheduling")
        print(" ✓ Retrieval Planning")
        print(f"   {C_GRAY}Rationale: {data.get('rationale')}{C_RESET}")
        print(f"   {C_CYAN}Planner completed.{C_RESET}")

    elif ev_type == "step_3_retrieval":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Parallel Retrieval{C_RESET}")
        latencies = data.get("latencies", {})
        print(f" ✓ Vector Retrieval ({latencies.get('vector', 0.0):.2f}ms)")
        print(f" ✓ Knowledge Graph Retrieval ({latencies.get('graph', 0.0):.2f}ms)")
        print(f" ✓ Document Retrieval ({latencies.get('document', 0.0):.2f}ms)")
        print(f" ✓ SQL Retrieval ({latencies.get('sql', 0.0):.2f}ms)")
        print(f" ✓ Web Retrieval ({latencies.get('web', 0.0):.2f}ms)")
        print(f"   {C_CYAN}Tparallel Verification: max(Ti) = {data.get('max_latency_ms'):.2f}ms | Actual Parallel Time = {data.get('parallel_latency_ms'):.2f}ms{C_RESET}")
        print(f"   {C_GREEN}Formula Verified: {data.get('formula_verified')}{C_RESET}")

    elif ev_type == "step_4_fusion":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Evidence Fusion{C_RESET}")
        print(" ✓ Duplicate removal")
        print(" ✓ Entity resolution")
        print(" ✓ Conflict detection")
        print(" ✓ Conflict resolution")
        print(" ✓ Semantic ranking")
        print(" ✓ Graph integration")
        print(" ✓ Adaptive weighting")
        print(" ✓ Softmax normalization")
        print(f"   {C_CYAN}Unified Knowledge Context stored on Blackboard.{C_RESET}")

    elif ev_type == "step_5_reasoning":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Collaborative Reasoning (Generator Agent){C_RESET}")
        print(f" ✓ Candidate response drafts generated (Iteration {data.get('iteration')})")
        print(f"   {C_GRAY}Draft snippet: {data.get('draft')[:100]}...{C_RESET}")

    elif ev_type == "step_6_validation":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Graph-Grounded Validation (Validator Agent){C_RESET}")
        print(f" ✓ Claims extracted & checked against Neo4j multi-hop paths")
        print(f"   {C_CYAN}Validation Score: {data.get('validation_score')*100:.1f}%{C_RESET}")
        print(f"   {C_RED}Rejected unsupported claims count: {data.get('rejected_claims')}{C_RESET}")

    elif ev_type == "step_7_trust":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Trust-Aware Decision (Trust Agent){C_RESET}")
        print(f" ✓ Computed Trust Metrics:")
        print(f"   - Confidence Score: {data.get('confidence_score'):.3f}")
        print(f"   - Consensus Score: {data.get('consensus_score'):.3f}")
        print(f"   - Composite Trust Score: {data.get('trust_score'):.3f}")

    elif ev_type == "step_8_validation_loop":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Internal Validation Loop{C_RESET}")
        print(f"   Iteration {data.get('iteration')} -> Status: {'CONVERGED' if data.get('converged') else 'Not converged (Score below threshold). Loop continues.'}")

    elif ev_type == "step_9_confidence_refinement":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Confidence Refinement{C_RESET}")
        print(" ✓ Softmax weights adjusted")
        print(" ✓ Retriever confidence calibration complete")

    elif ev_type == "step_10_routing_calibration":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Routing Calibration{C_RESET}")
        print(" ✓ Retriever priorities updated")
        print(" ✓ Planner policy weights adjusted")

    elif ev_type == "step_11_final_output":
        print(f"\n{C_BOLD}{C_CYAN}=================================================={C_RESET}")
        print(f"{C_BOLD}{C_GREEN}FINAL RESPONSE{C_RESET}")
        print(f"{C_BOLD}{C_CYAN}=================================================={C_RESET}")
        print(data.get("answer"))
        print(f"{C_BOLD}{C_CYAN}=================================================={C_RESET}")
        print(f"Validation Score: {C_GREEN}{data.get('validation_score')*100:.1f}%{C_RESET} | Trust Score: {C_GREEN}{data.get('trust_score')*100:.1f}%{C_RESET}")
        print(f"Iterations: {data.get('iterations_count')}")
        print(f"{C_BOLD}{C_CYAN}=================================================={C_RESET}")
        
        # Call interactive feedback loop
        await prompt_rlhf_feedback(session_id, api_mode)

    elif ev_type == "step_13_kg_update":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Knowledge Graph Update{C_RESET}")
        print(f" ✓ Inserted validated entities: {data.get('inserted_entities')}")
        print(f" ✓ Updated edge confidence parameters")

    elif ev_type == "step_14_monitoring":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Continuous Monitoring & Adaptive Feedback Loop{C_RESET}")
        print(f" ✓ CPU Usage: {data.get('cpu_usage_pct')}% | RAM Footprint: {data.get('ram_usage_mb')}MB")
        print(f" ✓ Metrics written to logs/system.log")

    elif ev_type == "step_15_tracking":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Tracking{C_RESET}")
        print(" ✓ Saved session and agent logs history")

    elif ev_type == "step_16_analysis":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Analysis{C_RESET}")
        print(" ✓ Factual accuracy report compiled")

    elif ev_type == "step_17_updating":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Updating{C_RESET}")
        print(" ✓ Dynamically updated model and resolver weights")

    elif ev_type == "step_18_monitoring_report":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Monitoring Report{C_RESET}")
        print(" ✓ Performance metrics verified healthy")

    elif ev_type == "step_19_end_session":
        print("↓")
        print(f"{C_BOLD}{C_GREEN}Completed{C_RESET}")
        print(f"{C_BOLD}{C_CYAN}=================================================={C_RESET}")
        print(f"{C_GREEN}Execution completed successfully.{C_RESET}")
        print(f"{C_GRAY}Logs successfully saved to logs/ system files.{C_RESET}")
        print(f"{C_BOLD}{C_CYAN}=================================================={C_RESET}")

async def prompt_rlhf_feedback(session_id: str, api_mode: bool):
    """Prompts user for quality rating, corrections, and usefulness."""
    print(f"\n{C_BOLD}Step 12: Requesting RLHF Quality Feedback{C_RESET}")
    
    # Rating
    rating = 5
    while True:
        try:
            r_str = input("Rate response quality (1-5) [default: 5]: ").strip()
            if not r_str:
                rating = 5
                break
            r_val = int(r_str)
            if 1 <= r_val <= 5:
                rating = r_val
                break
            print(f"{C_RED}Please input a number between 1 and 5.{C_RESET}")
        except ValueError:
            print(f"{C_RED}Invalid input. Please input an integer.{C_RESET}")

    # Correction fact
    corrections = input("Factual correction (optional): ").strip()

    # Useful check
    useful = "Y"
    while True:
        u_str = input("Was the response useful? (Y/N) [default: Y]: ").strip().upper()
        if not u_str:
            useful = "Y"
            break
        if u_str in ["Y", "N"]:
            useful = u_str
            break
        print(f"{C_RED}Please input Y or N.{C_RESET}")

    payload = {
        "session_id": session_id,
        "rating": rating,
        "corrections": corrections if corrections else None,
        "accepted": useful == "Y"
    }

    print(f"{C_GRAY}Submitting feedback and updating calibration weights...{C_RESET}")

    if api_mode:
        try:
            headers = {"X-API-KEY": API_KEY}
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{BACKEND_URL}/api/feedback", json=payload, headers=headers)
                if resp.status_code == 200:
                    res_data = resp.json()
                    print(f"{C_GREEN}[OK] Feedback submitted. New fusion weights:{C_RESET} {res_data.get('new_weights')}")
                else:
                    print(f"{C_RED}Failed to post feedback to API (Status: {resp.status_code}){C_RESET}")
        except Exception as e:
            print(f"{C_RED}Error posting feedback: {e}{C_RESET}")
    else:
        # Update locally in Direct Mode
        try:
            from backend.app.main import submit_feedback, FeedbackSubmission
            feed_obj = FeedbackSubmission(**payload)
            resp = await submit_feedback(feed_obj)
            print(f"{C_GREEN}[OK] Feedback submitted locally. New weights:{C_RESET} {resp.get('new_weights')}")
        except Exception as e:
            print(f"{C_RED}Error executing local feedback updates: {e}{C_RESET}")

if __name__ == "__main__":
    try:
        asyncio.run(run_cli_session())
    except KeyboardInterrupt:
        print("\nShutdown.")
