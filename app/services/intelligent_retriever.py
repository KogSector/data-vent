"""
Data Vent - Intelligent Retrieval Engine
Performs vector similarity search, DFS traversal, relevance scoring,
and completion checking against FalkorDB.
"""
import structlog
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import httpx
import json

logger = structlog.get_logger()


@dataclass
class ChunkNode:
    """A retrieved chunk with its metadata and scores."""
    chunk_id: str
    content: str
    score: float
    chunk_type: str = ""
    source_id: str = ""
    document_id: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
    depth: int = 0
    path: str = ""


@dataclass
class DFSTraversalResult:
    """Result of a DFS traversal."""
    chunks: List[ChunkNode]
    nodes_visited: int
    completion_reached: bool


class IntelligentRetriever:
    """
    Intelligent retrieval engine that combines vector similarity search
    with DFS graph traversal and relevance scoring.
    """
    
    def __init__(
        self,
        falcordb_uri: str,
        falcordb_username: str,
        falcordb_password: str,
        embeddings_service_url: str,
        vector_dimension: int = 384,
        similarity_threshold: float = 0.75,
        max_results: int = 100,
    ):
        self.falcordb_uri = falcordb_uri
        self.falcordb_username = falcordb_username
        self.falcordb_password = falcordb_password
        self.embeddings_service_url = embeddings_service_url
        self.vector_dimension = vector_dimension
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        self._driver = None
        self._http_client = None
    
    async def initialize(self):
        """Initialize connections to FalkorDB and embeddings service."""
        try:
            from neo4j import AsyncGraphDatabase
            self._driver = AsyncGraphDatabase.driver(
                self.falcordb_uri,
                auth=(self.falcordb_username, self.falcordb_password),
            )
            logger.info("intelligent_retriever_initialized", uri=self.falcordb_uri)
        except Exception as e:
            logger.error("falcordb_connection_failed", error=str(e))
            raise
        
        self._http_client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """Clean up connections."""
        if self._driver:
            await self._driver.close()
        if self._http_client:
            await self._http_client.aclose()
    
    async def vector_search(
        self,
        query_vectors: List[float],
        limit: int = 10,
        similarity_threshold: float = None,
        source_ids: List[str] = None,
        chunk_types: List[str] = None,
    ) -> List[ChunkNode]:
        """
        Perform vector similarity search in FalkorDB.
        Returns chunks ranked by similarity score.
        """
        threshold = similarity_threshold or self.similarity_threshold
        
        # Build Cypher query for vector similarity search
        filters = []
        params: Dict[str, Any] = {
            "query_vector": query_vectors,
            "limit": limit,
            "threshold": threshold,
        }
        
        if source_ids:
            filters.append("node.source_id IN $source_ids")
            params["source_ids"] = source_ids
        
        if chunk_types:
            filters.append("node.chunk_type IN $chunk_types")
            params["chunk_types"] = chunk_types
        
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        
        query = f"""
        CALL db.idx.vector.queryNodes('Vector_Chunk', 'embedding', $limit, vecf32($query_vector))
        YIELD node, score
        {where_clause}
        WHERE score >= $threshold
        RETURN node.chunk_id AS chunk_id,
               node.content AS content,
               score,
               node.chunk_type AS chunk_type,
               node.source_id AS source_id,
               node.document_id AS document_id
        ORDER BY score DESC
        LIMIT $limit
        """
        
        results = []
        try:
            async with self._driver.session() as session:
                result = await session.run(query, params)
                async for record in result:
                    results.append(ChunkNode(
                        chunk_id=record["chunk_id"],
                        content=record["content"],
                        score=record["score"],
                        chunk_type=record.get("chunk_type", ""),
                        source_id=record.get("source_id", ""),
                        document_id=record.get("document_id", ""),
                    ))
            
            logger.info("vector_search_completed", results=len(results), threshold=threshold)
        except Exception as e:
            logger.error("vector_search_failed", error=str(e))
        
        return results
    
    async def dfs_traversal(
        self,
        start_chunk_ids: List[str],
        max_depth: int = 3,
        min_relevance: float = 0.3,
        max_results: int = 50,
        query_context: Optional[Dict[str, str]] = None,
    ) -> DFSTraversalResult:
        """
        Perform DFS traversal from starting chunks.
        Propagates relevance scores through the graph.
        Uses completion flags to stop when enough context is retrieved.
        """
        visited = set()
        results = []
        nodes_visited = 0
        completion_reached = False
        
        # DFS traversal query with relevance propagation
        query = """
        MATCH path = (start:Vector_Chunk {chunk_id: $start_id})-[*1..$max_depth]-(related:Vector_Chunk)
        WHERE related.completion_flag = false
          AND related.chunk_id <> $start_id
        WITH related, 
             length(path) AS depth,
             reduce(score = 1.0, rel IN relationships(path) | score * COALESCE(rel.weight, 0.8)) AS path_score
        WHERE path_score >= $min_relevance
        RETURN DISTINCT related.chunk_id AS chunk_id,
               related.content AS content,
               path_score AS relevance_score,
               depth,
               related.source_id AS source_id,
               related.document_id AS document_id,
               related.chunk_type AS chunk_type
        ORDER BY relevance_score DESC
        LIMIT $max_results
        """
        
        try:
            async with self._driver.session() as session:
                for start_id in start_chunk_ids:
                    if len(results) >= max_results:
                        completion_reached = True
                        break
                    
                    result = await session.run(query, {
                        "start_id": start_id,
                        "max_depth": max_depth,
                        "min_relevance": min_relevance,
                        "max_results": max_results - len(results),
                    })
                    
                    async for record in result:
                        chunk_id = record["chunk_id"]
                        if chunk_id not in visited:
                            visited.add(chunk_id)
                            nodes_visited += 1
                            results.append(ChunkNode(
                                chunk_id=chunk_id,
                                content=record["content"],
                                score=record["relevance_score"],
                                depth=record["depth"],
                                source_id=record.get("source_id", ""),
                                document_id=record.get("document_id", ""),
                                chunk_type=record.get("chunk_type", ""),
                                path=json.dumps({"start": start_id, "depth": record["depth"]}),
                            ))
            
            # Check completion
            completion_reached = completion_reached or len(results) >= max_results
            
            logger.info(
                "dfs_traversal_completed",
                results=len(results),
                nodes_visited=nodes_visited,
                completion=completion_reached,
            )
        except Exception as e:
            logger.error("dfs_traversal_failed", error=str(e))
        
        return DFSTraversalResult(
            chunks=results,
            nodes_visited=nodes_visited,
            completion_reached=completion_reached,
        )
    
    async def hybrid_search(
        self,
        query_text: str,
        query_vectors: List[float],
        limit: int = 20,
        similarity_threshold: float = None,
        dfs_depth: int = 2,
        dfs_min_relevance: float = 0.3,
        source_ids: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Hybrid search: vector similarity + DFS graph traversal.
        1. Vector search to find top matches
        2. DFS from top matches to find related chunks
        3. Merge and rank results
        """
        import time
        start_time = time.time()
        
        # Step 1: Vector similarity search
        vector_results = await self.vector_search(
            query_vectors=query_vectors,
            limit=limit // 2,
            similarity_threshold=similarity_threshold,
            source_ids=source_ids,
        )
        
        # Step 2: DFS traversal from top vector matches
        start_ids = [r.chunk_id for r in vector_results[:5]]  # Top 5 as seeds
        dfs_result = DFSTraversalResult(chunks=[], nodes_visited=0, completion_reached=False)
        if start_ids:
            dfs_result = await self.dfs_traversal(
                start_chunk_ids=start_ids,
                max_depth=dfs_depth,
                min_relevance=dfs_min_relevance,
                max_results=limit // 2,
            )
        
        # Step 3: Merge and deduplicate
        seen = set()
        merged = []
        for chunk in vector_results:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                merged.append(chunk)
        for chunk in dfs_result.chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                merged.append(chunk)
        
        # Sort by score
        merged.sort(key=lambda c: c.score, reverse=True)
        merged = merged[:limit]
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "chunks": merged,
            "vector_matches": len(vector_results),
            "graph_matches": len(dfs_result.chunks),
            "completion_reached": dfs_result.completion_reached,
            "total_time_ms": total_time,
        }
    
    async def vectorize_query(self, query_text: str) -> List[float]:
        """Call embeddings service to vectorize a query string."""
        try:
            response = await self._http_client.post(
                f"{self.embeddings_service_url}/api/v1/generate",
                json={"text": query_text},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embeddings", [])
        except Exception as e:
            logger.error("query_vectorization_failed", error=str(e))
            return []
    
    async def relevance_scoring(
        self,
        chunks: List[ChunkNode],
        query: str,
    ) -> List[ChunkNode]:
        """Re-rank chunks by relevance to the query."""
        # Simple text overlap scoring as fallback
        query_words = set(query.lower().split())
        for chunk in chunks:
            content_words = set(chunk.content.lower().split())
            overlap = len(query_words & content_words) / max(len(query_words), 1)
            # Boost the embedding score with text overlap
            chunk.score = chunk.score * 0.8 + overlap * 0.2
        
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks
    
    async def completion_check(
        self,
        retrieved_chunks: List[ChunkNode],
        requirements: Dict[str, Any],
    ) -> bool:
        """
        Check if enough context has been retrieved.
        Uses heuristics based on content coverage and diversity.
        """
        min_chunks = requirements.get("min_chunks", 3)
        max_chunks = requirements.get("max_chunks", 50)
        min_score = requirements.get("min_avg_score", 0.5)
        
        if len(retrieved_chunks) < min_chunks:
            return False
        
        if len(retrieved_chunks) >= max_chunks:
            return True
        
        # Check average score
        avg_score = sum(c.score for c in retrieved_chunks) / len(retrieved_chunks)
        if avg_score >= min_score:
            return True
        
        return False
