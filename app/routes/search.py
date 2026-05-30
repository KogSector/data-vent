"""
Data Vent — Search Routes

Search endpoints backed by Graphify (vector + graph retrieval).
all search is now delegated to IntelligentRetriever → GraphifyService.
"""

from datetime import datetime
from typing import Any, Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.main import get_retriever

logger = structlog.get_logger()
router = APIRouter(prefix="/search", tags=["search"])


# ── Request / Response models ─────────────────────────────────────────────────

class SemanticSearchRequest(BaseModel):
    """Request model for Graphify semantic search."""
    query: str = Field(..., description="Natural language query")
    limit: int = Field(default=10, ge=1, le=200, description="Maximum results")
    group_ids: Optional[list[str]] = Field(default=None, description="Restrict to graph groups / source IDs")
    center_node_uuid: Optional[str] = Field(default=None, description="Rerank around this entity node UUID")


class SearchResultItem(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any]
    source: str


class SemanticSearchResponse(BaseModel):
    results: list[SearchResultItem]
    total_count: int
    search_time_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/semantic", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest):
    """
    Semantic search using Graphify hybrid vector + graph retrieval.

    Optionally rerank results around a known entity node (center_node_uuid)
    for graph-aware context propagation.
    """
    import time
    start = time.perf_counter()

    retriever = get_retriever()
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        results = await retriever.retrieve(
            query=request.query,
            group_ids=request.group_ids,
            num_results=request.limit,
            center_node_uuid=request.center_node_uuid,
        )
    except Exception as exc:
        logger.error("semantic_search_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    elapsed_ms = (time.perf_counter() - start) * 1000
    return SemanticSearchResponse(
        results=[SearchResultItem(**r.to_dict()) for r in results],
        total_count=len(results),
        search_time_ms=round(elapsed_ms, 2),
    )


@router.get("/health")
async def search_health_check():
    """Health check for the search layer."""
    retriever = get_retriever()
    return {
        "status": "healthy" if retriever else "initialising",
        "service": "search",
        "active_backend": retriever.active_backend if retriever else "none",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/")
async def search_info():
    """Get information about search capabilities."""
    retriever = get_retriever()
    return {
        "name": "Data-Vent Search Service",
        "version": "3.0.0",
        "active_backend": retriever.active_backend if retriever else "none",
        "backends": {
            "graphify": "Hybrid vector + graph (new)",
        },
        "capabilities": [
            "Hybrid vector + graph search (Graphify)",
            "Graph-aware entity-centred reranking",
            "Temporal knowledge graph queries",
        ],
        "endpoints": [
            {"path": "/search/semantic", "method": "POST", "description": "Semantic search"},
            {"path": "/search/health", "method": "GET", "description": "Health check"},
            {"path": "/search/", "method": "GET", "description": "Service information"},
        ],
    }