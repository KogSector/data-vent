"""
Data Vent - Main Application Entry Point
Intelligent retrieval engine with HTTP + gRPC servers.
Kafka consumer: listens to chunks.stored notifications to update search index.
"""
import asyncio
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import settings
from app.services.intelligent_retriever import IntelligentRetriever

logger = structlog.get_logger()


# Global state
_retriever: IntelligentRetriever = None


def get_retriever() -> IntelligentRetriever:
    """Get the global retriever instance."""
    return _retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager — initialize and cleanup services."""
    global _retriever
    
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
        await start_grpc_server(_retriever)
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


# Create FastAPI app
app = FastAPI(
    title="Data Vent - Intelligent Retrieval Engine",
    description="Semantic search, DFS traversal, and graph queries using FalkorDB",
    version="0.1.0",
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


# Health endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "data-vent",
        "version": "0.1.0",
        "ports": {
            "http": settings.PORT,
            "grpc": settings.GRPC_PORT,
        },
    }


# Search endpoints
@app.post("/api/v1/search")
async def search(request: dict):
    """Vector similarity search."""
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
    """Hybrid search (vector + graph traversal)."""
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
