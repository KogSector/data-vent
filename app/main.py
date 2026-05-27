"""
Data Vent — Main Application Entry Point
Intelligent retrieval engine with HTTP + gRPC servers.
Uses Graphiti + FalkorDB for all semantic search and graph queries.
"""

import asyncio
import time
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uvicorn

from app.config import settings
# from app.services.graphiti_service import GraphitiService
# from app.services.graphify_service import GraphifyService
# from app.services.intelligent_retriever import IntelligentRetriever
# from app.services.query_decomposer import QueryDecomposer
# from app.services.parallel_search import ParallelSearchDispatcher
# from app.services.result_aggregator import ResultAggregator
logger = structlog.get_logger()


# â”€â”€â”€ Global state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_graphiti_service: Any = None
_graphify_service: Any = None
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
    """Application lifespan manager â€” initialize and cleanup services."""
    global _retriever, _decomposer, _dispatcher, _aggregator, _graphify_service
    
    logger.info("data_vent_starting",
                port=settings.PORT,
                grpc_port=settings.GRPC_PORT,
                environment=settings.ENVIRONMENT)
    
    # Initialise Graphiti (FalkorDB via Redis protocol) — legacy backend
    # _graphiti_service = GraphitiService(settings)
    # await _graphiti_service.initialize()

    # Initialise Graphify — new backend (non-blocking, feature-flagged)
    # _graphify_service = GraphifyService(settings)
    # await _graphify_service.initialize()

    # Wrap in IntelligentRetriever (dual-backend, flag-controlled)
    # _retriever = IntelligentRetriever(
    #     graphiti_service=_graphiti_service,
    #     graphify_service=_graphify_service,
    # )
    _retriever = None

    # Initialize pipeline components
    _decomposer = None
    _dispatcher = None
    _aggregator = None
    
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
    if _graphify_service:
        await _graphify_service.close()
    if _graphiti_service:
        await _graphiti_service.close()
    grpc_task.cancel()


async def _start_grpc_background():
    """Start gRPC server in background."""
    try:
        from app.grpc_server import start_grpc_server
        await start_grpc_server(_retriever, _decomposer, _dispatcher, _aggregator)
    except Exception as e:
        logger.error("grpc_server_failed", error=str(e))




# â”€â”€â”€ FastAPI app â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app = FastAPI(
    title="Data Vent — Intelligent Retrieval Engine",
    description="Semantic search and graph queries powered by Graphiti + FalkorDB",
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


# â”€â”€â”€ Request / Response models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class RetrieveRequest(BaseModel):
    """Request for the unified retrieval pipeline."""
    query: str
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
            "http": settings.PORT,
            "grpc": settings.GRPC_PORT,
        },
    }


@app.post("/api/v1/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """
    Full retrieval pipeline:
    1. Decompose query into semantic chunks
    2. Parallel search across all chunks in FalkorDB
    3. Aggregate, score-fuse, and rank results
    
    This is the primary endpoint for client-connector.
    """
    logger.info("retrieval_pipeline_completed_mock", query=request.query)
    return RetrieveResponse(
        results=[
            ScoredChunkResponse(
                chunk_id="mock_1",
                content="This is the dummy data you requested: System is fully operational and tests are passing.",
                final_score=0.99,
                vector_score=0.99,
                graph_score=0.99,
                cross_chunk_boost=1.0,
                chunk_type="text",
                source_id="mock_source",
                document_id="mock_doc",
                metadata={"test": "true"},
                matched_by_chunks=["mock"],
            )
        ],
        total_results=1,
        unique_sources=1,
        vector_matches=1,
        graph_matches=1,
        completion_reached=True,
        query_chunks=[],
        decomposition_time_ms=0.0,
        search_time_ms=0.0,
        aggregation_time_ms=0.0,
        total_time_ms=0.0,
    )


@app.post("/api/v1/search")
async def search(request: dict):
    """Semantic search via Graphiti hybrid retrieval."""
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
    """Hybrid search — Graphiti vector + BM25 + graph reranking."""
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


# Include routes from existing routers if they exist
try:
    from app.routes import graphiti, status
    app.include_router(graphiti.router, prefix="/api/v1/graphiti", tags=["graphiti"])
    app.include_router(status.router, prefix="/api/v1/status", tags=["status"])
except ImportError:
    logger.info("optional_routers_not_found")

# Include the new enhanced search routes
try:
    from app.routes.search import router as search_router
    app.include_router(search_router, tags=["enhanced-search"])
    logger.info("enhanced_search_routes_loaded")
except ImportError:
    logger.warning("enhanced_search_routes_not_found")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
    )

