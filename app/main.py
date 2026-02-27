"""
Data Vent - Main Application
Semantic search and graph traversal service using Graphiti and FalcorDB
"""
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import httpx

from .config import settings
from .services.graphiti_service import GraphitiService
from .services.feature_toggle_client import FeatureToggleClient
from .routes import search, health

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("data_vent_starting", port=settings.PORT)
    
    # Initialize services
    app.state.http_client = httpx.AsyncClient(timeout=30.0)
    app.state.feature_toggle_client = FeatureToggleClient(
        base_url=settings.FEATURE_TOGGLE_SERVICE_URL,
        http_client=app.state.http_client
    )
    app.state.graphiti_service = GraphitiService(
        falcordb_uri=settings.FALCORDB_URI,
        falcordb_username=settings.FALCORDB_USERNAME,
        falcordb_password=settings.FALCORDB_PASSWORD,
        embeddings_service_url=settings.EMBEDDINGS_SERVICE_URL,
        feature_toggle_client=app.state.feature_toggle_client,
        http_client=app.state.http_client
    )
    
    await app.state.graphiti_service.initialize()
    
    logger.info("data_vent_started", 
                services=["graphiti", "falcordb", "embeddings"],
                llm_enabled=await app.state.feature_toggle_client.is_enabled("enableLLM"))
    
    yield
    
    # Cleanup
    logger.info("data_vent_shutting_down")
    await app.state.graphiti_service.close()
    await app.state.http_client.aclose()
    logger.info("data_vent_stopped")


# Create FastAPI app
app = FastAPI(
    title="Data Vent Service",
    description="Semantic search and graph traversal using Graphiti and FalcorDB",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(search.router, prefix="/api/search", tags=["search"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "data-vent",
        "version": "0.1.0",
        "description": "Semantic search and graph traversal service",
        "endpoints": {
            "health": "/health",
            "search": "/api/search",
            "semantic_search": "/api/search/semantic",
            "graph_traverse": "/api/search/graph",
            "hybrid_search": "/api/search/hybrid"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development"
    )
