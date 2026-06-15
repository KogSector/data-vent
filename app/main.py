"""
Data Vent — Main Application Entry Point
Intelligent retrieval engine with HTTP + gRPC servers.
Uses Graphify + FalkorDB for semantic search and graph queries.
"""

import asyncio
import time
import uuid
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uvicorn

from app.config import settings
from app.services.vector_search import build_client_from_settings
from app.services.intelligent_retriever import IntelligentRetriever
from app.services.query_decomposer import QueryDecomposer, QueryChunk
from app.services.parallel_search import ParallelSearchDispatcher
from app.services.result_aggregator import ResultAggregator

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# ─── Global state ────────────────────────────────────────────────────────────────

_retriever: Any = None
_decomposer: Any = None
_dispatcher: Any = None
_aggregator: Any = None


def get_retriever() -> Any:
    """Get the global retriever instance."""
    return _retriever


def get_pipeline():
    """Get the full pipeline components."""
    return _decomposer, _dispatcher, _aggregator, _retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager — initialize and cleanup services."""
    global _retriever, _decomposer, _dispatcher, _aggregator
    
    logger.info("data_vent_starting",
                port=settings.APP_PORT,
                grpc_port=settings.GRPC_PORT,
                environment=settings.ENVIRONMENT)
    
    # Initialise FalkorDB
    falkordb_client = build_client_from_settings(settings)
    await falkordb_client.connect()

    # Wrap in IntelligentRetriever
    _retriever = IntelligentRetriever(falkordb_client=falkordb_client, settings=settings)

    # Initialize pipeline components
    _decomposer = QueryDecomposer(max_chunks=settings.PIPELINE_MAX_QUERY_CHUNKS)
    _dispatcher = ParallelSearchDispatcher(
        per_chunk_timeout=settings.PIPELINE_PER_CHUNK_TIMEOUT,
        vector_top_k=settings.PIPELINE_VECTOR_TOP_K,
        dfs_depth=settings.PIPELINE_DFS_DEPTH,
        dfs_min_relevance=settings.PIPELINE_DFS_MIN_RELEVANCE,
        dfs_max_results=settings.PIPELINE_DFS_MAX_RESULTS,
    )
    _aggregator = ResultAggregator(
        max_results=settings.PIPELINE_MAX_TOTAL_RESULTS,
        vector_weight=settings.PIPELINE_VECTOR_WEIGHT,
        graph_weight=settings.PIPELINE_GRAPH_WEIGHT,
        cross_chunk_weight=settings.PIPELINE_CROSS_CHUNK_WEIGHT,
    )
    
    logger.info("retrieval_pipeline_initialized",
                max_chunks=settings.PIPELINE_MAX_QUERY_CHUNKS,
                vector_top_k=settings.PIPELINE_VECTOR_TOP_K,
                dfs_depth=settings.PIPELINE_DFS_DEPTH)
    
    # Start gRPC server in background
    grpc_task = asyncio.create_task(_start_grpc_background())
    
    logger.info("data_vent_started", status="ready")
    
    yield
    
    # Cleanup
    logger.info("data_vent_shutting_down")
    await _retriever.close()
    await falkordb_client.close()
    grpc_task.cancel()


async def _start_grpc_background():
    """Start gRPC server in background."""
    try:
        from app.grpc_server import start_grpc_server
        await start_grpc_server(_retriever, _decomposer, _dispatcher, _aggregator)
    except Exception as e:
        logger.error("grpc_server_failed", error=str(e))

app = FastAPI(
    title="Data Vent — Intelligent Retrieval Engine",
    description="Semantic search and graph queries powered by FalkorDB",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RetrieveRequest(BaseModel):
    """Request for the unified retrieval pipeline."""
    intent: str
    keywords: List[str]
    limit: int = Field(default=20, ge=1, le=200)
    source_ids: Optional[List[str]] = None
    options: Optional[Dict[str, str]] = None


class ScoredChunkResponse(BaseModel):
    """A single scored chunk in the response."""
    chunk_id: str
    content: str
    final_score: float
    vector_score: float
    graph_score: float
    cross_chunk_boost: float
    chunk_type: str = ""
    source_id: str = ""
    document_id: str = ""
    metadata: Dict[str, str] = {}
    matched_by_chunks: List[str] = []


class QueryChunkResponse(BaseModel):
    """Info about a decomposed query chunk."""
    text: str
    intent: str
    weight: float


class RetrieveResponse(BaseModel):
    """Response from the unified retrieval pipeline."""
    results: List[ScoredChunkResponse]
    total_results: int
    unique_sources: int
    vector_matches: int
    graph_matches: int
    completion_reached: bool
    query_chunks: List[QueryChunkResponse]
    decomposition_time_ms: float
    search_time_ms: float
    aggregation_time_ms: float
    total_time_ms: float


# â”€â”€â”€ Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "data-vent",
        "version": "0.2.0",
        "pipeline": "active",
        "ports": {
            "http": settings.APP_PORT,
            "grpc": settings.GRPC_PORT,
        },
    }


@app.post("/api/v1/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest, req: Request):
    """
    Full retrieval pipeline:
    1. Decompose query into semantic chunks
    2. Parallel search across all chunks in FalkorDB
    3. Aggregate, score-fuse, and rank results
    
    This is the primary endpoint for client-connector.
    """
    if not _retriever or not _decomposer or not _dispatcher or not _aggregator:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Pipeline components not initialized")

    request_id = req.headers.get("x-request-id", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(request_id=request_id)
    logger.info("retrieval_request_received", intent=request.intent, keywords=request.keywords, limit=request.limit, source_ids=request.source_ids)

    start_time = time.perf_counter()
    
    # 1. Decompose Intent
    decomp_result = await _decomposer.decompose(request.intent)
    
    # Generate explicit chunks from keywords
    keyword_chunks = [
        QueryChunk(
            text=kw,
            intent="entity_lookup",
            weight=1.0,
            original_span=(0, 0),
            tokens=kw.split()
        )
        for kw in request.keywords if kw.strip()
    ]
    
    # Combine chunks
    all_chunks = keyword_chunks + decomp_result.chunks
    
    logger.info("query_decomposed", chunks_count=len(all_chunks), decomposition_time_ms=decomp_result.decomposition_time_ms)
    
    # 2. Parallel Search
    search_result = await _dispatcher.dispatch(all_chunks, _retriever)
    logger.info("parallel_search_completed", search_time_ms=search_result.total_time_ms)
    
    # 3. Aggregate
    agg_result = await _aggregator.aggregate(search_result, original_query=request.intent, limit=request.limit)
    logger.info("results_aggregated", aggregation_time_ms=agg_result.aggregation_time_ms)
    
    total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    
    results = [
        ScoredChunkResponse(
            chunk_id=c.chunk_id,
            content=c.content,
            final_score=c.final_score,
            vector_score=c.vector_score,
            graph_score=c.graph_score,
            cross_chunk_boost=c.cross_chunk_boost,
            chunk_type=c.chunk_type,
            source_id=c.source_id,
            document_id=c.document_id,
            metadata=c.metadata,
            matched_by_chunks=c.matched_by_chunks,
        ) for c in agg_result.chunks
    ]
    
    query_chunks = [
        QueryChunkResponse(text=c.text, intent=c.intent, weight=c.weight)
        for c in all_chunks
    ]
    
    logger.info("retrieval_pipeline_completed", intent=request.intent, total_results=len(results), total_time_ms=total_time_ms)
    
    return RetrieveResponse(
        results=results,
        total_results=agg_result.total_results,
        unique_sources=agg_result.unique_sources,
        vector_matches=agg_result.vector_matches,
        graph_matches=agg_result.graph_matches,
        completion_reached=agg_result.completion_reached,
        query_chunks=query_chunks,
        decomposition_time_ms=decomp_result.decomposition_time_ms,
        search_time_ms=search_result.total_time_ms,
        aggregation_time_ms=agg_result.aggregation_time_ms,
        total_time_ms=total_time_ms,
    )


@app.post("/api/v1/search")
async def search(request: dict):
    """Semantic search via IntelligentRetriever."""
    retriever = get_retriever()
    if not retriever:
        return {"error": "Retriever not initialized"}, 503

    query_text = request.get("query", "")
    limit = request.get("limit", 10)
    source_ids = request.get("source_ids")

    results = await retriever.retrieve(
        query=query_text,
        group_ids=source_ids,
        num_results=limit,
    )

    return {
        "chunks": [r.to_dict() for r in results],
        "total": len(results),
    }


@app.post("/api/v1/hybrid-search")
async def hybrid_search(request: dict):
    """Hybrid search — vector + BM25 + graph reranking."""
    retriever = get_retriever()
    if not retriever:
        return {"error": "Retriever not initialized"}, 503

    query_text = request.get("query", "")
    limit = request.get("limit", 20)
    source_ids = request.get("source_ids")
    center_node_uuid = request.get("center_node_uuid")

    results = await retriever.retrieve(
        query=query_text,
        group_ids=source_ids,
        num_results=limit,
        center_node_uuid=center_node_uuid,
    )

    return {
        "chunks": [r.to_dict() for r in results],
        "total": len(results),
    }


try:
    from app.routes import status
    app.include_router(status.router, prefix="/api/v1/status", tags=["status"])
except ImportError:
    logger.info("optional_routers_not_found")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.APP_PORT,
        reload=settings.ENVIRONMENT == "development",
    )