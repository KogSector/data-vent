"""
Data Vent — Intelligent Retriever

Retrieval engine that routes queries to FalkorDB via vector and graph search.
"""

from __future__ import annotations

import json
from typing import Any, List

import httpx
import structlog
from dataclasses import dataclass, field

from app.services.vector_search import FalkorDBClient

logger = structlog.get_logger(__name__)

@dataclass
class SearchResult:
    """Normalised retrieval result."""
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any]
    source: str = "falkordb"
    chunk_type: str = ""
    source_id: str = ""
    document_id: str = ""
    depth: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source,
            "chunk_type": self.chunk_type,
            "source_id": self.source_id,
            "document_id": self.document_id,
            "depth": self.depth,
        }

@dataclass
class DFSTraversalResult:
    chunks: List[SearchResult]

class IntelligentRetriever:
    """
    Retrieval engine using FalkorDBClient and embeddings service.
    """

    def __init__(self, falkordb_client: FalkorDBClient, settings: Any) -> None:
        self._falkordb = falkordb_client
        self.ollama_url = getattr(settings, "OLLAMA_URL", "http://localhost:11434")
        self._http_client = httpx.AsyncClient(timeout=15.0)

    @property
    def active_backend(self) -> str:
        """Return the name of the currently active retrieval backend."""
        return "falkordb"

    async def close(self) -> None:
        await self._http_client.aclose()

    async def vectorize_query(self, query: str) -> List[float]:
        """Generate embeddings for a single query string using Ollama directly."""
        try:
            response = await self._http_client.post(
                f"{self.ollama_url}/api/embed",
                json={"input": [query], "model": "nomic-embed-text"}
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [])
            return embeddings[0] if embeddings else []
        except Exception as e:
            logger.error("vectorize_query_failed", query=query[:80], error=str(e))
            return []

    async def vector_search(self, query_vectors: List[float], limit: int = 10) -> List[SearchResult]:
        """Perform vector similarity search on FalkorDB."""
        if not query_vectors:
            return []
            
        try:
            embedding_str = json.dumps(query_vectors)
            cypher = f"""
            CALL db.idx.vector.queryNodes('Vector_Chunk', 'embeddings', {limit}, vecf32({embedding_str})) YIELD node, score
            RETURN node.id AS chunk_id,
                   node.content AS content,
                   node.chunk_type AS chunk_type,
                   node.source_id AS source_id,
                   node.metadata AS metadata,
                   score AS score
            ORDER BY score DESC
            """
            raw_result = await self._falkordb.query(cypher)
            return self._parse_graph_results(raw_result, is_vector=True)
        except Exception as e:
            logger.error("vector_search_failed", error=str(e))
            return []

    async def text_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Perform Full-Text Search on FalkorDB as a fallback."""
        try:
            import re
            words = re.findall(r'\b\w+\b', query.lower())
            keywords = [w for w in words if len(w) > 2]
            if not keywords and words:
                keywords = words
                
            if not keywords:
                return []
                
            fts_query = " ".join(keywords)
            
            cypher = f"""
            CALL db.idx.fulltext.queryNodes('Vector_Chunk', '{fts_query}') YIELD node, score
            RETURN node.id AS chunk_id,
                   node.content AS content,
                   node.chunk_type AS chunk_type,
                   node.source_id AS source_id,
                   node.metadata AS metadata,
                   score AS score
            ORDER BY score DESC
            LIMIT {limit}
            """
            raw_result = await self._falkordb.query(cypher)
            return self._parse_graph_results(raw_result, is_vector=True)
        except Exception as e:
            logger.error("text_search_failed", error=str(e))
            return []

    async def dfs_traversal(
        self,
        start_chunk_ids: List[str],
        max_depth: int = 2,
        min_relevance: float = 0.3,
        max_results: int = 20,
    ) -> DFSTraversalResult:
        """Perform variable-length path traversal from start nodes in FalkorDB."""
        if not start_chunk_ids:
            return DFSTraversalResult(chunks=[])
            
        try:
            # We construct a Cypher query using an IN clause for the start nodes
            ids_str = ", ".join(f"'{cid}'" for cid in start_chunk_ids)
            
            # Using variable length path. We cap the depth.
            cypher = f"""
            MATCH path = (start:Vector_Chunk)-[*1..{max_depth}]-(n:Vector_Chunk)
            WHERE start.id IN [{ids_str}] AND NOT n.id IN [{ids_str}]
            RETURN DISTINCT n.id AS chunk_id,
                   n.content AS content,
                   n.chunk_type AS chunk_type,
                   n.source_id AS source_id,
                   n.metadata AS metadata,
                   length(path) AS depth
            LIMIT {max_results}
            """
            raw_result = await self._falkordb.query(cypher)
            chunks = self._parse_graph_results(raw_result, is_vector=False)
            
            # Base graph score decreases with depth.
            for chunk in chunks:
                chunk.score = max(0.1, 1.0 - (chunk.depth * 0.2))
                
            # Filter by relevance
            chunks = [c for c in chunks if c.score >= min_relevance]
            
            return DFSTraversalResult(chunks=chunks)
            
        except Exception as e:
            logger.error("dfs_traversal_failed", error=str(e))
            return DFSTraversalResult(chunks=[])

    def _parse_graph_results(self, raw_result: Any, is_vector: bool) -> List[SearchResult]:
        """Parse raw GRAPH.QUERY output into SearchResult objects."""
        # FalkorDB typically returns [[headers], [[row1], [row2]], [stats]]
        if not raw_result or len(raw_result) < 2:
            return []
            
        headers_raw = raw_result[0]
        rows_raw = raw_result[1]
        
        # Decode headers
        headers = []
        for h in headers_raw:
            if isinstance(h, bytes):
                headers.append(h.decode('utf-8'))
            else:
                headers.append(str(h))
                
        results = []
        for row in rows_raw:
            chunk_id = ""
            content = ""
            chunk_type = ""
            source_id = ""
            metadata_str = "{}"
            score = 0.0
            depth = 0
            
            for i, val in enumerate(row):
                if i >= len(headers):
                    continue
                    
                col_name = headers[i]
                
                # Extract value
                v = val
                if isinstance(val, bytes):
                    v = val.decode('utf-8')
                    
                if col_name == "chunk_id":
                    chunk_id = str(v)
                elif col_name == "content":
                    content = str(v)
                elif col_name == "chunk_type":
                    chunk_type = str(v)
                elif col_name == "source_id":
                    source_id = str(v)
                elif col_name == "metadata":
                    metadata_str = str(v) if v else "{}"
                elif col_name == "score" and is_vector:
                    try:
                        score = float(v)
                    except (ValueError, TypeError):
                        pass
                elif col_name == "depth":
                    try:
                        depth = int(v)
                    except (ValueError, TypeError):
                        pass
                        
            metadata = {}
            try:
                if metadata_str and metadata_str != "None":
                    metadata = json.loads(metadata_str)
            except json.JSONDecodeError:
                pass
                
            results.append(SearchResult(
                chunk_id=chunk_id,
                content=content,
                score=score,
                metadata=metadata,
                chunk_type=chunk_type,
                source_id=source_id,
                document_id=source_id,
                depth=depth,
            ))
            
        return results

    async def retrieve(
        self,
        query: str,
        group_ids: list[str] | None = None,
        num_results: int = 10,
        center_node_uuid: str | None = None,
    ) -> list[SearchResult]:
        vector = await self.vectorize_query(query)
        results = []
        if vector:
            results = await self.vector_search(vector, num_results)
            
        # Fallback to text search if vector search returns 0 results (e.g. embeddings service was down)
        if not results:
            logger.info("Vector search returned 0 results, falling back to text search", query=query)
            results = await self.text_search(query, num_results)
            
        return results

    async def retrieve_with_context(
        self,
        query: str,
        entity_uuid: str,
        group_ids: list[str] | None = None,
        num_results: int = 15,
    ) -> list[SearchResult]:
        return await self.retrieve(query, group_ids, num_results, center_node_uuid=entity_uuid)
