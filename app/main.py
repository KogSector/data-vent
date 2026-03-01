"""
Data Vent - Main Application Entry Point
Intelligent retrieval engine with HTTP + gRPC servers.
Kafka consumer: listens to chunks.stored notifications to update search index.
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
from app.services.intelligent_retriever import IntelligentRetriever
from app.services.query_decomposer import QueryDecomposer
from app.services.parallel_search import ParallelSearchDispatcher
from app.services.result_aggregator import ResultAggregator

logger = structlog.get_logger()


# â”€â”€â”€ Global state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_retriever: IntelligentRetriever = None
_decomposer: QueryDecomposer = None
_dispatcher: ParallelSearchDispatcher = None
_aggregator: ResultAggregator = None


def get_retriever() -> IntelligentRetriever:
    """Get the global retriever instance."""
    return _retriever


def get_pipeline():
    """Get the full pipeline components."""
    return _decomposer, _dispatcher, _aggregator, _retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager â€” initialize and cleanup services."""
    global _retriever, _decomposer, _dispatcher, _aggregator
    
    logger.info("data_vent_starting",
                port=settings.PORT,
                grpc_port=settings.GRPC_PORT,
                environment=settings.ENVIRONMENT)
    
    # Initialize intelligent retriever
    _retriever = IntelligentRetriever(
        falcordb_uri=settings.FALCORDB_URI,
        falcordb_username=settings.FALCORDB_USERNAME,
        falcordb_password=settings.FALCORDB_PASSWORD,
        embeddings_service_url=settings.EMBEDDINGS_SERVICE_URL,
        vector_dimension=settings.FALCORDB_VECTOR_DIMENSION,
        similarity_threshold=settings.FALCORDB_SIMILARITY_THRESHOLD,
        max_results=settings.FALCORDB_MAX_RESULTS,
    )
    await _retriever.initialize()
    
    # Initialize pipeline components
    _decomposer = QueryDecomposer(
        max_chunks=settings.PIPELINE_MAX_QUERY_CHUNKS,
    )
    
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
    
    # Start Kafka consumer for chunks.stored notifications
    kafka_task = asyncio.create_task(_start_kafka_consumer())
    
    logger.info("data_vent_started", status="ready")
    
    yield
    
    # Cleanup
    logger.info("data_vent_shutting_down")
    if _retriever:
        await _retriever.close()
    grpc_task.cancel()
    kafka_task.cancel()


async def _start_grpc_background():
    """Start gRPC server in background."""
    try:
        from app.grpc_server import start_grpc_server
        await start_grpc_server(_retriever, _decomposer, _dispatcher, _aggregator)
    except Exception as e:
        logger.error("grpc_server_failed", error=str(e))


async def _start_kafka_consumer():
    """Start Kafka consumer for chunks.stored notifications."""
    try:
        from aiokafka import AIOKafkaConsumer
        import json
        
        bootstrap_servers = settings.KAFKA_BOOTSTRAP_SERVERS
        consumer = AIOKafkaConsumer(
            "chunks.stored",
            bootstrap_servers=bootstrap_servers,
            group_id="data-vent",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=False,
        )
        await consumer.start()
        logger.info("Kafka consumer started", topic="chunks.stored", bootstrap_servers=bootstrap_servers)
        
        try:
            async for msg in consumer:
                try:
                    notification = msg.value
                    chunk_ids = notification.get("chunk_ids", [])
                    source_id = notification.get("source_id", "unknown")
                    
                    logger.info(
                        "chunks_stored_notification",
                        chunk_ids_count=len(chunk_ids),
                        source_id=source_id,
                    )
                    
                    # Index update is handled by FalkorDB directly
                    # data-vent reads from FalkorDB on query time
                    # This notification primarily signals that new data is available
                    
                    await consumer.commit()
                except Exception as e:
                    logger.error("kafka_message_processing_failed", error=str(e))
        finally:
            await consumer.stop()
    except Exception as e:
        logger.error("kafka_consumer_failed", error=str(e))


# â”€â”€â”€ FastAPI app â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

app = FastAPI(
    title="Data Vent - Intelligent Retrieval Engine",
    description="Semantic search, DFS traversal, and graph queries using FalkorDB",
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
    pipeline_start = time.perf_counter()
    
    decomposer, dispatcher, aggregator, retriever = get_pipeline()
    if not retriever:
        return {"error": "Retriever not initialized"}, 503
    
    # Step 1: Decompose
    decomposition = await decomposer.decompose(request.query)
    
    # Step 2: Parallel search
    parallel_result = await dispatcher.dispatch(
        chunks=decomposition.chunks,
        retriever=retriever,
    )
    
    # Step 3: Aggregate
    aggregated = await aggregator.aggregate(
        parallel_result=parallel_result,
        original_query=request.query,
        limit=request.limit,
    )
    
    total_time = (time.perf_counter() - pipeline_start) * 1000
    
    logger.info(
        "retrieval_pipeline_completed",
        query=request.query[:80],
        chunks_decomposed=decomposition.total_chunks,
        results=aggregated.total_results,
        total_time_ms=round(total_time, 2),
    )
    
    return RetrieveResponse(
        results=[
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
            )
            for c in aggregated.chunks
        ],
        total_results=aggregated.total_results,
        unique_sources=aggregated.unique_sources,
        vector_matches=aggregated.vector_matches,
        graph_matches=aggregated.graph_matches,
        completion_reached=aggregated.completion_reached,
        query_chunks=[
            QueryChunkResponse(
                text=qc.text,
                intent=qc.intent,
                weight=qc.weight,
            )
            for qc in decomposition.chunks
        ],
        decomposition_time_ms=round(decomposition.decomposition_time_ms, 2),
        search_time_ms=round(parallel_result.total_time_ms, 2),
        aggregation_time_ms=round(aggregated.aggregation_time_ms, 2),
        total_time_ms=round(total_time, 2),
    )


@app.post("/api/v1/search")
async def search(request: dict):
    """Vector similarity search (legacy endpoint)."""
    retriever = get_retriever()
    if not retriever:
        return {"error": "Retriever not initialized"}, 503
    
    query_text = request.get("query", "")
    limit = request.get("limit", 10)
    source_ids = request.get("source_ids")
    
    # Vectorize query
    query_vectors = await retriever.vectorize_query(query_text)
    if not query_vectors:
        return {"error": "Failed to vectorize query"}, 500
    
    results = await retriever.vector_search(
        query_vectors=query_vectors,
        limit=limit,
        source_ids=source_ids,
    )
    
    return {
        "chunks": [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "score": r.score,
                "chunk_type": r.chunk_type,
                "source_id": r.source_id,
            }
            for r in results
        ],
        "total": len(results),
    }


@app.post("/api/v1/hybrid-search")
async def hybrid_search(request: dict):
    """Hybrid search â€” vector + graph traversal (legacy endpoint)."""
    retriever = get_retriever()
    if not retriever:
        return {"error": "Retriever not initialized"}, 503
    
    query_text = request.get("query", "")
    limit = request.get("limit", 20)
    dfs_depth = request.get("dfs_depth", 2)
    source_ids = request.get("source_ids")
    
    # Vectorize query
    query_vectors = await retriever.vectorize_query(query_text)
    if not query_vectors:
        return {"error": "Failed to vectorize query"}, 500
    
    result = await retriever.hybrid_search(
        query_text=query_text,
        query_vectors=query_vectors,
        limit=limit,
        dfs_depth=dfs_depth,
        source_ids=source_ids,
    )
    
    return {
        "chunks": [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "score": r.score,
                "chunk_type": r.chunk_type,
                "source_id": r.source_id,
            }
            for r in result["chunks"]
        ],
        "vector_matches": result["vector_matches"],
        "graph_matches": result["graph_matches"],
        "completion_reached": result["completion_reached"],
        "total_time_ms": result["total_time_ms"],
    }


# Include routes from existing routers if they exist
try:
    from app.routes import graphiti, status
    app.include_router(graphiti.router, prefix="/api/v1/graphiti", tags=["graphiti"])
    app.include_router(status.router, prefix="/api/v1/status", tags=["status"])
except ImportError:
    logger.info("optional_routers_not_found")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
    )

