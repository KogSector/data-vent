"""
Data Vent - Search Routes
Enhanced search endpoints with vector similarity and hybrid search capabilities.
"""

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime
import structlog

from app.services.vector_search import (
    VectorSearchService, 
    FalcorDBConfig, 
    SearchFilters, 
    GraphTraversalParams,
    HybridSearchResult,
    VectorSearchResult
)
from app.services.query_decomposer import QueryDecomposer, QueryChunk
from app.services.parallel_search import ParallelSearchDispatcher
from app.services.result_aggregator import ResultAggregator

logger = structlog.get_logger()
router = APIRouter(prefix="/search", tags=["search"])

# Global service instances
_vector_search_service: Optional[VectorSearchService] = None
_query_decomposer: Optional[QueryDecomposer] = None
_parallel_search: Optional[ParallelSearchDispatcher] = None
_result_aggregator: Optional[ResultAggregator] = None


def get_vector_search_service() -> VectorSearchService:
    """Get the global vector search service instance."""
    global _vector_search_service
    if _vector_search_service is None:
        config = FalcorDBConfig.from_env()
        _vector_search_service = VectorSearchService(config)
    return _vector_search_service


def get_search_pipeline():
    """Get the complete search pipeline components."""
    global _query_decomposer, _parallel_search, _result_aggregator
    return _query_decomposer, _parallel_search, _result_aggregator


# Pydantic models for API requests/responses
class VectorSearchRequest(BaseModel):
    """Request model for vector similarity search."""
    query_vector: List[float] = Field(..., description="Query vector for similarity search")
    limit: int = Field(default=10, ge=1, le=1000, description="Maximum number of results")
    threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="Similarity threshold")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Search filters")


class VectorSearchResponse(BaseModel):
    """Response model for vector similarity search."""
    results: List[Dict[str, Any]]
    total_count: int
    search_time_ms: float
    query_metadata: Dict[str, Any]


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search."""
    query_vector: List[float] = Field(..., description="Query vector for similarity search")
    graph_params: Optional[Dict[str, Any]] = Field(default=None, description="Graph traversal parameters")
    limit: int = Field(default=10, ge=1, le=1000, description="Maximum number of results")


class HybridSearchResponse(BaseModel):
    """Response model for hybrid search."""
    results: List[Dict[str, Any]]
    total_count: int
    search_time_ms: float
    graph_metadata: Dict[str, Any]


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search with query decomposition."""
    query: str = Field(..., description="Natural language query")
    limit: int = Field(default=10, ge=1, le=1000, description="Maximum number of results")
    use_parallel_search: bool = Field(default=True, description="Use parallel search decomposition")
    search_mode: str = Field(default="hybrid", regex="^(vector|graph|hybrid)$", description="Search mode")


class SemanticSearchResponse(BaseModel):
    """Response model for semantic search."""
    results: List[Dict[str, Any]]
    total_count: int
    search_time_ms: float
    query_chunks: List[Dict[str, Any]]
    search_metadata: Dict[str, Any]


@router.post("/vector", response_model=VectorSearchResponse)
async def vector_similarity_search(request: VectorSearchRequest):
    """
    Perform vector similarity search against FalcorDB.
    
    This endpoint provides direct vector similarity search capabilities,
    finding chunks with similar vector embeddings to the query vector.
    """
    import time
    start_time = time.time()
    
    try:
        service = get_vector_search_service()
        
        # Parse filters if provided
        filters = None
        if request.filters:
            filters = SearchFilters(
                document_ids=[uuid.UUID(doc_id) for doc_id in request.filters.get("document_ids", [])] 
                    if request.filters.get("document_ids") else None,
                source_ids=request.filters.get("source_ids"),
                workspace_id=request.filters.get("workspace_id"),
                content_type=request.filters.get("content_type"),
                date_range=tuple(
                    datetime.fromisoformat(date_str) 
                    for date_str in request.filters.get("date_range", [])
                ) if request.filters.get("date_range") else None,
            )
        
        # Perform vector search
        results = await service.similarity_search(
            query_vector=request.query_vector,
            limit=request.limit,
            threshold=request.threshold,
            filters=filters,
        )
        
        # Convert results to dicts for JSON serialization
        result_dicts = []
        for result in results:
            result_dict = {
                "chunk_id": str(result.chunk_id),
                "chunk_text": result.chunk_text,
                "document_id": str(result.document_id),
                "source_id": result.source_id,
                "similarity_score": result.similarity_score,
                "chunk_index": result.chunk_index,
                "metadata": result.metadata,
            }
            result_dicts.append(result_dict)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return VectorSearchResponse(
            results=result_dicts,
            total_count=len(result_dicts),
            search_time_ms=elapsed_ms,
            query_metadata={
                "vector_dimension": len(request.query_vector),
                "threshold": request.threshold,
                "filters_applied": filters is not None,
            }
        )
        
    except Exception as e:
        logger.error("vector_search_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")


@router.post("/hybrid", response_model=HybridSearchResponse)
async def hybrid_search(request: HybridSearchRequest):
    """
    Perform hybrid search combining vector similarity with graph traversal.
    
    This endpoint provides comprehensive search by combining vector similarity
    with graph-based relationship traversal for more contextual results.
    """
    import time
    start_time = time.time()
    
    try:
        service = get_vector_search_service()
        
        # Parse graph traversal parameters
        graph_params = GraphTraversalParams()
        if request.graph_params:
            graph_params.relationship_types = request.graph_params.get("relationship_types", graph_params.relationship_types)
            graph_params.max_depth = request.graph_params.get("max_depth", graph_params.max_depth)
            graph_params.node_filters = request.graph_params.get("node_filters")
            graph_params.relationship_weights = request.graph_params.get("relationship_weights")
        
        # Perform hybrid search
        results = await service.hybrid_search(
            query_vector=request.query_vector,
            graph_params=graph_params,
            limit=request.limit,
        )
        
        # Convert results to dicts for JSON serialization
        result_dicts = []
        for result in results:
            result_dict = {
                "vector_result": {
                    "chunk_id": str(result.vector_result.chunk_id),
                    "chunk_text": result.vector_result.chunk_text,
                    "document_id": str(result.vector_result.document_id),
                    "source_id": result.vector_result.source_id,
                    "similarity_score": result.vector_result.similarity_score,
                    "chunk_index": result.vector_result.chunk_index,
                    "metadata": result.vector_result.metadata,
                },
                "related_chunks": [
                    {
                        "chunk_id": str(rc.chunk_id),
                        "relationship_type": rc.relationship_type,
                        "relationship_score": rc.relationship_score,
                    }
                    for rc in result.related_chunks
                ],
                "entities": [
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "mention_count": entity.mention_count,
                    }
                    for entity in result.entities
                ],
                "combined_score": result.combined_score,
            }
            result_dicts.append(result_dict)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return HybridSearchResponse(
            results=result_dicts,
            total_count=len(result_dicts),
            search_time_ms=elapsed_ms,
            graph_metadata={
                "relationship_types": graph_params.relationship_types,
                "max_depth": graph_params.max_depth,
                "relationship_weights": graph_params.relationship_weights,
            }
        )
        
    except Exception as e:
        logger.error("hybrid_search_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@router.post("/semantic", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest):
    """
    Perform semantic search with query decomposition and parallel processing.
    
    This endpoint accepts natural language queries, decomposes them into semantic chunks,
    and performs parallel search across multiple query components for comprehensive results.
    """
    import time
    start_time = time.time()
    
    try:
        # Get search pipeline components
        decomposer, dispatcher, aggregator = get_search_pipeline()
        
        if not all([decomposer, dispatcher, aggregator]):
            # Initialize pipeline components if not already done
            from app.main import get_pipeline
            decomposer, dispatcher, aggregator, _ = get_pipeline()
        
        # Decompose the query
        query_chunks = decomposer.decompose_query(request.query)
        
        chunk_dicts = [
            {
                "text": chunk.text,
                "weight": chunk.weight,
                "intent": chunk.intent,
                "entities": chunk.entities,
            }
            for chunk in query_chunks
        ]
        
        if request.use_parallel_search and len(query_chunks) > 1:
            # Use parallel search for complex queries
            parallel_result = await dispatcher.search_chunks(query_chunks, request.search_mode)
            final_results = aggregator.aggregate_results(parallel_result, request.limit)
        else:
            # Use simple search for single chunks or when parallel is disabled
            if query_chunks:
                service = get_vector_search_service()
                if request.search_mode == "vector":
                    # Get embeddings for the query (this would call embeddings-service)
                    # For now, we'll simulate this
                    vector_results = await service.similarity_search(
                        query_vector=[0.1] * 384,  # Placeholder vector
                        limit=request.limit,
                    )
                    final_results = [{"vector_result": vr, "related_chunks": [], "entities": [], "combined_score": vr.similarity_score} for vr in vector_results]
                elif request.search_mode == "hybrid":
                    hybrid_results = await service.hybrid_search(
                        query_vector=[0.1] * 384,  # Placeholder vector
                        limit=request.limit,
                    )
                    final_results = hybrid_results
                else:
                    final_results = []
            else:
                final_results = []
        
        # Convert results to dicts for JSON serialization
        result_dicts = []
        for result in final_results:
            if hasattr(result, 'vector_result'):
                # Hybrid search result
                result_dict = {
                    "vector_result": {
                        "chunk_id": str(result.vector_result.chunk_id),
                        "chunk_text": result.vector_result.chunk_text,
                        "document_id": str(result.vector_result.document_id),
                        "source_id": result.vector_result.source_id,
                        "similarity_score": result.vector_result.similarity_score,
                        "chunk_index": result.vector_result.chunk_index,
                        "metadata": result.vector_result.metadata,
                    },
                    "related_chunks": [
                        {
                            "chunk_id": str(rc.chunk_id),
                            "relationship_type": rc.relationship_type,
                            "relationship_score": rc.relationship_score,
                        }
                        for rc in result.related_chunks
                    ],
                    "entities": [
                        {
                            "id": entity.id,
                            "name": entity.name,
                            "entity_type": entity.entity_type,
                            "mention_count": entity.mention_count,
                        }
                        for entity in result.entities
                    ],
                    "combined_score": result.combined_score,
                }
            else:
                # Simple vector result
                result_dict = {
                    "chunk_id": str(result.chunk_id),
                    "chunk_text": result.chunk_text,
                    "document_id": str(result.document_id),
                    "source_id": result.source_id,
                    "similarity_score": result.similarity_score,
                    "chunk_index": result.chunk_index,
                    "metadata": result.metadata,
                }
            result_dicts.append(result_dict)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return SemanticSearchResponse(
            results=result_dicts,
            total_count=len(result_dicts),
            search_time_ms=elapsed_ms,
            query_chunks=chunk_dicts,
            search_metadata={
                "search_mode": request.search_mode,
                "use_parallel_search": request.use_parallel_search,
                "chunks_processed": len(query_chunks),
            }
        )
        
    except Exception as e:
        logger.error("semantic_search_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@router.get("/health")
async def search_health_check():
    """Health check for search services."""
    try:
        service = get_vector_search_service()
        is_healthy = await service.health_check()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "vector_search",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error("search_health_check_failed", error=str(e))
        return {
            "status": "unhealthy",
            "service": "vector_search",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@router.get("/")
async def search_info():
    """Get information about search capabilities."""
    return {
        "name": "Data-Vent Search Service",
        "version": "1.0.0",
        "capabilities": [
            "Vector similarity search",
            "Hybrid search with graph traversal",
            "Semantic search with query decomposition",
            "Parallel search processing",
            "Multi-modal content support",
        ],
        "endpoints": [
            {"path": "/search/vector", "method": "POST", "description": "Vector similarity search"},
            {"path": "/search/hybrid", "method": "POST", "description": "Hybrid search with graph traversal"},
            {"path": "/search/semantic", "method": "POST", "description": "Semantic search with decomposition"},
            {"path": "/search/health", "method": "GET", "description": "Health check"},
            {"path": "/search/", "method": "GET", "description": "Service information"},
        ],
        "configuration": {
            "vector_dimension": FalcorDBConfig.from_env().vector_dimension,
            "similarity_threshold": FalcorDBConfig.from_env().similarity_threshold,
            "max_results": FalcorDBConfig.from_env().max_results,
        }
    }