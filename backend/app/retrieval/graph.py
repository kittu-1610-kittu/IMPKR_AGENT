import re
from typing import List
from backend.app.retrieval.base import BaseRetriever
from backend.app.database.connections import graph_mgr
from shared.schema import Evidence

class GraphRetriever(BaseRetriever):
    def __init__(self):
        super().__init__(name="GraphRetriever")

    async def _retrieve_impl(self, query: str, session_id: str, limit: int) -> List[Evidence]:
        # Identify core entity keywords
        keywords = ["orchestrator", "blackboard", "planner", "generator", "critic", "validator", "trust", "fusion", "retrieval"]
        found_entities = []
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', query.lower()):
                found_entities.append(kw)

        if not found_entities:
            words = [w.strip("?,.!") for w in query.split()]
            found_entities = [w for w in words if len(w) > 4][:2]

        evidence_list = []
        
        for entity in found_entities:
            if not graph_mgr.is_mock:
                # Real Neo4j Multi-Hop Path Query (Table 9 Depth 1 to 4 Hops)
                cypher = """
                MATCH p = (n)-[r*1..4]->(m)
                WHERE (toLower(n.name) CONTAINS toLower($entity_name)
                   OR toLower(m.name) CONTAINS toLower($entity_name))
                  AND all(rel in r WHERE rel.confidence >= 0.45)
                RETURN p
                LIMIT $limit
                """
                records = await graph_mgr.execute_cypher(cypher, {"entity_name": entity, "limit": limit})
                
                for idx, record in enumerate(records):
                    path = record.get("p")
                    if path:
                        nodes = [node["name"] for node in path.nodes]
                        rels = [rel.type for rel in path.relationships]
                        
                        path_str = f" -> ".join([f"({n})" for n in nodes])
                        rel_str = ", ".join(rels)
                        content = f"Knowledge Graph path found: {path_str} via relationships [{rel_str}]."
                        
                        evidence_list.append(
                            Evidence(
                                id=f"graph_{session_id}_{entity}_hop_{idx}",
                                content=content,
                                source_type="graph",
                                confidence=0.95,
                                metadata={
                                    "nodes": nodes,
                                    "relationships": rels,
                                    "hops": len(rels)
                                }
                            )
                        )
            else:
                # InMemoryGraph Mock Multi-Hop BFS query (Table 9 Depth 1 to 4 Hops)
                db = graph_mgr.mock_db
                if not db:
                    continue
                
                # Find start nodes matching entity
                start_nids = []
                for nid, val in db.nodes.items():
                    name = val["properties"].get("name", "").lower()
                    if entity.lower() in name or nid.lower() == entity.lower():
                        start_nids.append(nid)
                
                hop_idx = 0
                for start_nid in start_nids:
                    src_name = db.nodes[start_nid]["properties"].get("name", start_nid)
                    
                    # Queue items: (current_node_id, path_node_names, path_relations_types, current_confidence)
                    queue = [(start_nid, [src_name], [], 1.0)]
                    
                    while queue:
                        curr_nid, path_nodes, path_rels, curr_conf = queue.pop(0)
                        depth = len(path_rels)
                        
                        # Add paths of depth 1..4 as evidence
                        if 1 <= depth <= 4:
                            path_str = " -> ".join([f"({n})" for n in path_nodes])
                            rel_str = ", ".join(path_rels)
                            content = f"Knowledge Graph path found: {path_str} via relationships [{rel_str}]."
                            
                            evidence_list.append(
                                Evidence(
                                    id=f"graph_{session_id}_{entity}_hop_{depth}_{hop_idx}",
                                    content=content,
                                    source_type="graph",
                                    confidence=curr_conf,
                                    metadata={
                                        "nodes": path_nodes,
                                        "relationships": path_rels,
                                        "hops": depth
                                    }
                                )
                            )
                            hop_idx += 1
                            
                        if depth >= 4:
                            continue
                            
                        # Retrieve outward edges
                        child_edges = [edge for edge in db.edges if edge["source"] == curr_nid]
                        
                        # Table 9: filter by Edge Weight Connectivity >= 0.45
                        child_edges = [edge for edge in child_edges if edge.get("confidence", 1.0) >= 0.45]
                        
                        # Table 9: cap neighbor expansion at 12
                        child_edges = child_edges[:12]
                        
                        for edge in child_edges:
                            target = edge["target"]
                            if target not in db.nodes:
                                continue
                            target_name = db.nodes[target]["properties"].get("name", target)
                            # Avoid loop revisits in active path
                            if target_name not in path_nodes:
                                queue.append((
                                    target,
                                    path_nodes + [target_name],
                                    path_rels + [edge["type"]],
                                    curr_conf * edge["confidence"]
                                ))

        # Deduplicate
        seen_content = set()
        dedup_list = []
        for ev in evidence_list:
            if ev.content not in seen_content:
                seen_content.add(ev.content)
                dedup_list.append(ev)

        return dedup_list[:limit]
