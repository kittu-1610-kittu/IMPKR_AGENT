import logging
from backend.app.database.connections import db_mgr

logger = logging.getLogger("DatabaseSeeder")

async def seed_relational_database():
    logger.info("Checking relational database schema...")
    
    # Create tables
    queries = [
        """
        CREATE TABLE IF NOT EXISTS system_components (
            id VARCHAR(50) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            type VARCHAR(50) NOT NULL,
            description TEXT,
            status VARCHAR(20) DEFAULT 'active'
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS validation_rules (
            rule_id VARCHAR(50) PRIMARY KEY,
            target_component VARCHAR(50) REFERENCES system_components(id),
            rule_name VARCHAR(100) NOT NULL,
            constraint_type VARCHAR(50) NOT NULL,
            min_score FLOAT DEFAULT 0.0
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS performance_metrics (
            metric_id SERIAL PRIMARY KEY,
            component_id VARCHAR(50) REFERENCES system_components(id),
            latency_ms FLOAT,
            success_rate FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    ]

    for q in queries:
        await db_mgr.execute_query(q)

    # Check if data already exists
    rows = await db_mgr.execute_query("SELECT COUNT(*) as count FROM system_components;")
    count = rows[0]["count"] if rows else 0

    if count == 0:
        logger.info("Database is empty. Seeding mock system_components...")
        
        insert_components = [
            ("AgentOrchestrator", "Agent Orchestrator", "Core", "Coordinates multi-agent flow and blackboard updates.", "active"),
            ("Blackboard", "Shared Blackboard", "Memory", "Central communication repository for agents.", "active"),
            ("Planner", "Planner Agent", "Agent", "Decomposes queries and routes retrieval.", "active"),
            ("Generator", "Generator Agent", "Agent", "Generates source-grounded responses.", "active"),
            ("Critic", "Critic Agent", "Agent", "Reviews drafts for logical gaps.", "active"),
            ("Validator", "Validator Agent", "Agent", "Factual verification against data sources.", "active"),
            ("Trust", "Trust Agent", "Agent", "Computes consensus metrics and signals convergence.", "active"),
            ("ParallelRetrieval", "Parallel Retrieval", "System", "Simultaneously queries Vector, Graph, PostgreSQL and Web.", "active"),
            ("EvidenceFusion", "Adaptive Evidence Fusion", "System", "Applies Softmax weights to score and fuse retrieved facts.", "active")
        ]

        for cid, name, ctype, desc, status in insert_components:
            await db_mgr.execute_query(
                "INSERT INTO system_components (id, name, type, description, status) VALUES ($1, $2, $3, $4, $5);",
                cid, name, ctype, desc, status
            )

        insert_rules = [
            ("rule_planning", "Planner", "Valid Plan Structure", "schema", 0.90),
            ("rule_generation", "Generator", "Grounded Claims Only", "semantic", 0.95),
            ("rule_critique", "Critic", "Identify Gaps", "logical", 0.85),
            ("rule_validation", "Validator", "Graph-Grounded Triples", "graph", 0.90),
            ("rule_trust", "Trust", "Consensus Score", "numeric", 0.85)
        ]

        for rid, cid, name, ctype, score in insert_rules:
            await db_mgr.execute_query(
                "INSERT INTO validation_rules (rule_id, target_component, rule_name, constraint_type, min_score) VALUES ($1, $2, $3, $4, $5);",
                rid, cid, name, ctype, score
            )
            
        logger.info("Successfully seeded system_components and validation_rules.")
    else:
        logger.info("Relational database already contains seeded data.")
