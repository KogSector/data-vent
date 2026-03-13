"""
Data Vent - Vector Search Service
Enhanced vector similarity search and hybrid search capabilities
ported from unified-processor Rust implementation.
"""

import structlog
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import asyncio
import httpx
from neo4j import AsyncGraphDatabase
import uuid

logger = structlog.get_logger()


@dataclass
class FalcorDBConfig:
    """Configuration for FalcorDB connection and vector operations."""
    host: str = "localhost"
    port: int = 6379
    username: str = "default"
    password: str = ""
    database: str = "default"
    vector_dimension: int = 384
    similarity_threshold: float = 0.75
    max_results: int = 100
    connection_pool_size: int = 20
    connection_timeout_ms: int = 5000
    query_timeout_ms: int = 30000

    @classmethod
    def from_env(cls) -> 'FalcorDBConfig':
        """Load configuration from environment variables."""
        import os
        return cls(
            host=os.getenv("FALCORDB_HOST", "localhost"),
            port=int(os.getenv("FALCORDB_PORT", "6379")),
            username=os.getenv("FALCORDB_USERNAME", "default"),
            password=os.getenv("FALCORDB_PASSWORD", ""),
            database=os.getenv("FALCORDB_DATABASE", "default"),
            vector_dimension=int(os.getenv("FALCORDB_VECTOR_DIMENSION", "384")),
            similarity_threshold=float(os.getenv("FALCORDB_SIMILARITY_THRESHOLD", "0.75")),
            max_results=int(os.getenv("FALCORDB_MAX_RESULTS", "100")),
            connection_pool_size=int(os.getenv("FALCORDB_CONNECTION_POOL_SIZE", "20")),
            connection_timeout_ms=int(os.getenv("FALCORDB_CONNECTION_TIMEOUT_MS", "5000")),
            query_timeout_ms=int(os.getenv("FALCORDB_QUERY_TIMEOUT_MS", "30000")),
        )


@dataclass
class SearchFilters:
    """Search filters for vector similarity queries."""
    document_ids: Optional[List[uuid.UUID]] = None
    source_ids: Optional[List[str]] = None
    workspace_id: Optional[str] = None
    content_type: Optional[str] = None
    date_range: Optional[Tuple[datetime, datetime]] = None


@dataclass
class DateRange:
    """Date range filter for temporal queries."""
    start: datetime
    end: datetime


@dataclass
class VectorSearchResult:
    """Vector similarity search result."""
    chunk_id: uuid.UUID
    chunk_text: str
    document_id: uuid.UUID
    source_id: str
    similarity_score: float
    chunk_index: int
    metadata: Dict[str, Any]


@dataclass
class RelatedChunk:
    """Related chunk found through graph traversal."""
    chunk_id: uuid.UUID
    relationship_type: str
    relationship_score: float


@dataclass
class Entity:
    """Entity extracted from chunk."""
    id: str
    name: str
    entity_type: str
    mention_count: int


@dataclass
class HybridSearchResult:
    """Hybrid search result combining vector similarity and graph traversal."""
    vector_result: VectorSearchResult
    related_chunks: List[RelatedChunk]
    entities: List[Entity]
    combined_score: float


@dataclass
class GraphTraversalParams:
    """Graph traversal parameters for hybrid search."""
    relationship_types: List[str] = field(default_factory=lambda: [
        "SIMILAR_TO", "RELATED_TO", "NEXT_CHUNK"
    ])
    max_depth: int = 2
    node_filters: Optional[Dict[str, Any]] = None
    relationship_weights: Optional[Dict[str, float]] = None


class FalcorDBClient:
    """FalcorDB client wrapper for connection management."""
    
    def __init__(self, config: FalcorDBConfig):
        self.config = config
        self._driver = None
        self._uri = f"bolt://{config.host}:{config.port}"
    
    async def connect(self):
        """Connect to FalcorDB with retry logic."""
        max_retries = 5
        initial_backoff_ms = 100
        retry_count = 0
        backoff_ms = initial_backoff_ms

        while retry_count < max_retries:
            try:
                self._driver = AsyncGraphDatabase.driver(
                    self._uri,
                    auth=(self.config.username, self.config.password),
                    max_connection_pool_size=self.config.connection_pool_size,
                    connection_timeout=self.config.connection_timeout_ms / 1000.0,
                )
                # Test connection
                await self._driver.verify_connectivity()
                logger.info("falcordb_connected", uri=self._uri)
                return
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception(f"Failed to connect to FalcorDB after {max_retries} retries: {e}")
                await asyncio.sleep(backoff_ms / 1000.0)
                backoff_ms = min(backoff_ms * 2, 5000)
    
    async def health_check(self) -> bool:
        """Health check to verify FalcorDB connectivity."""
        if not self._driver:
            return False
        try:
            async with self._driver.session(database=self.config.database) as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                return record is not None
        except Exception as e:
            logger.error("falcordb_health_check_failed", error=str(e))
            return False
    
    async def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        if not self._driver:
            raise Exception("Not connected to FalcorDB")
        
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records
    
    async def close(self):
        """Close the database connection."""
        if self._driver:
            await self._driver.close()


class VectorSearchService:
    """Vector search service with FalcorDB integration."""
    
    def __init__(self, config: FalcorDBConfig):
        self.config = config
        self.client = FalcorDBClient(config)
    
    async def initialize(self):
        """Initialize the FalcorDB connection."""
        await self.client.connect()
    
    async def similarity_search(
        self,
        query_vector: List[float],
        limit: int = None,
        threshold: float = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[VectorSearchResult]:
        """
        Perform vector similarity search in FalcorDB.
        """
        import time
        start_time = time.time()
        
        limit = limit or self.config.max_results
        threshold = threshold or self.config.similarity_threshold
        
        if len(query_vector) != self.config.vector_dimension:
            raise ValueError(f"Invalid query vector dimension: expected {self.config.vector_dimension}, got {len(query_vector)}")
        
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Invalid similarity threshold: must be between 0.0 and 1.0")
        
        # Build Cypher query for vector similarity search
        query_parts = [
            "CALL db.index.vector.queryNodes('vector_chunk_embedding', $limit, $query_vector)",
            "YIELD node, score",
            "WHERE score >= $threshold"
        ]
        
        # Add filters if provided
        if filters:
            if filters.document_ids:
                doc_ids = [str(doc_id) for doc_id in filters.document_ids]
                query_parts.append(f"AND node.document_id IN $document_ids")
            
            if filters.source_ids:
                query_parts.append(f"AND node.source_id IN $source_ids")
            
            if filters.workspace_id:
                query_parts.append(f"AND node.workspace_id = $workspace_id")
            
            if filters.content_type:
                query_parts.append(f"AND node.content_type = $content_type")
            
            if filters.date_range:
                query_parts.append(
                    "AND node.created_at >= datetime($start_date) "
                    "AND node.created_at <= datetime($end_date)"
                )
        
        query_parts.extend([
            "RETURN node.id as chunk_id,",
            "       node.chunk_text as chunk_text,",
            "       node.document_id as document_id,",
            "       node.source_id as source_id,",
            "       node.chunk_index as chunk_index,",
            "       node.metadata as metadata,",
            "       score as similarity_score",
            "ORDER BY score DESC",
            "LIMIT $limit"
        ])
        
        query = "\n".join(query_parts)
        
        # Prepare parameters
        parameters = {
            "query_vector": query_vector,
            "limit": limit,
            "threshold": threshold,
        }
        
        if filters:
            if filters.document_ids:
                parameters["document_ids"] = [str(doc_id) for doc_id in filters.document_ids]
            if filters.source_ids:
                parameters["source_ids"] = filters.source_ids
            if filters.workspace_id:
                parameters["workspace_id"] = filters.workspace_id
            if filters.content_type:
                parameters["content_type"] = filters.content_type
            if filters.date_range:
                parameters["start_date"] = filters.date_range[0].isoformat()
                parameters["end_date"] = filters.date_range[1].isoformat()
        
        try:
            records = await self.client.execute_query(query, parameters)
            
            results = []
            for record in records:
                try:
                    chunk_id = uuid.UUID(record["chunk_id"])
                    document_id = uuid.UUID(record["document_id"])
                    metadata = json.loads(record["metadata"]) if isinstance(record["metadata"], str) else record["metadata"]
                    
                    result = VectorSearchResult(
                        chunk_id=chunk_id,
                        chunk_text=record["chunk_text"],
                        document_id=document_id,
                        source_id=record["source_id"],
                        similarity_score=float(record["similarity_score"]),
                        chunk_index=int(record["chunk_index"]),
                        metadata=metadata,
                    )
                    results.append(result)
                except (ValueError, KeyError, json.JSONDecodeError) as e:
                    logger.warning("failed_to_parse_search_result", record=record, error=str(e))
                    continue
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                "vector_similarity_search_completed",
                results_count=len(results),
                elapsed_ms=elapsed_ms
            )
            
            return results
            
        except Exception as e:
            logger.error("vector_similarity_search_failed", error=str(e))
            raise
    
    async def hybrid_search(
        self,
        query_vector: List[float],
        graph_params: GraphTraversalParams = None,
        limit: int = None,
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining vector similarity and graph traversal.
        """
        import time
        start_time = time.time()
        
        graph_params = graph_params or GraphTraversalParams()
        limit = limit or self.config.max_results
        
        # Get vector similarity results (get more for better graph traversal)
        vector_results = await self.similarity_search(
            query_vector, 
            limit * 2, 
            self.config.similarity_threshold,
            None
        )
        
        if not vector_results:
            return []
        
        hybrid_results = []
        
        # For each vector result, perform graph traversal
        for vector_result in vector_results:
            related_chunks = await self._get_related_chunks(
                vector_result.chunk_id,
                graph_params.relationship_types,
                graph_params.max_depth
            )
            
            entities = await self._get_chunk_entities(vector_result.chunk_id)
            
            graph_score = self._calculate_graph_score(
                related_chunks, 
                graph_params.relationship_weights
            )
            
            combined_score = (vector_result.similarity_score * 0.7) + (graph_score * 0.3)
            
            hybrid_result = HybridSearchResult(
                vector_result=vector_result,
                related_chunks=related_chunks,
                entities=entities,
                combined_score=combined_score,
            )
            hybrid_results.append(hybrid_result)
        
        # Sort by combined score and limit results
        hybrid_results.sort(key=lambda x: x.combined_score, reverse=True)
        hybrid_results = hybrid_results[:limit]
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "hybrid_search_completed",
            results_count=len(hybrid_results),
            elapsed_ms=elapsed_ms
        )
        
        return hybrid_results
    
    async def _get_related_chunks(
        self,
        chunk_id: uuid.UUID,
        relationship_types: List[str],
        max_depth: int,
    ) -> List[RelatedChunk]:
        """Get related chunks through graph traversal."""
        query = f"""
        MATCH (vc:Vector_Chunk {{id: $chunk_id}})
        -[r:{'|'.join(relationship_types)}*1..{max_depth}]
        -(related:Vector_Chunk)
        RETURN related.id as chunk_id, 
               type(r[0]) as relationship_type,
               1.0 / size(r) as relationship_score
        LIMIT 50
        """
        
        parameters = {"chunk_id": str(chunk_id)}
        
        try:
            records = await self.client.execute_query(query, parameters)
            
            related_chunks = []
            for record in records:
                try:
                    chunk_id = uuid.UUID(record["chunk_id"])
                    related_chunk = RelatedChunk(
                        chunk_id=chunk_id,
                        relationship_type=record["relationship_type"],
                        relationship_score=float(record["relationship_score"]),
                    )
                    related_chunks.append(related_chunk)
                except (ValueError, KeyError) as e:
                    logger.warning("failed_to_parse_related_chunk", record=record, error=str(e))
                    continue
            
            return related_chunks
            
        except Exception as e:
            logger.error("get_related_chunks_failed", chunk_id=str(chunk_id), error=str(e))
            return []
    
    async def _get_chunk_entities(self, chunk_id: uuid.UUID) -> List[Entity]:
        """Get entities associated with a chunk."""
        query = """
        MATCH (vc:Vector_Chunk {id: $chunk_id})-[r:CONTAINS_ENTITY]->(e:Entity)
        RETURN e.id as id, 
               e.name as name, 
               e.type as entity_type,
               r.mention_count as mention_count
        LIMIT 20
        """
        
        parameters = {"chunk_id": str(chunk_id)}
        
        try:
            records = await self.client.execute_query(query, parameters)
            
            entities = []
            for record in records:
                try:
                    entity = Entity(
                        id=record["id"],
                        name=record["name"],
                        entity_type=record["entity_type"],
                        mention_count=int(record["mention_count"]),
                    )
                    entities.append(entity)
                except (ValueError, KeyError) as e:
                    logger.warning("failed_to_parse_entity", record=record, error=str(e))
                    continue
            
            return entities
            
        except Exception as e:
            logger.error("get_chunk_entities_failed", chunk_id=str(chunk_id), error=str(e))
            return []
    
    def _calculate_graph_score(
        self,
        related_chunks: List[RelatedChunk],
        relationship_weights: Optional[Dict[str, float]],
    ) -> float:
        """Calculate graph score based on related chunks and weights."""
        if not related_chunks:
            return 0.0
        
        total_score = 0.0
        for chunk in related_chunks:
            weight = 1.0
            if relationship_weights:
                weight = relationship_weights.get(chunk.relationship_type, 1.0)
            total_score += chunk.relationship_score * weight
        
        return total_score / len(related_chunks)
    
    async def health_check(self) -> bool:
        """Perform health check on the vector search service."""
        return await self.client.health_check()
    
    async def close(self):
        """Close the vector search service."""
        await self.client.close()
