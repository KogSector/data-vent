"""
Health check endpoints
"""
from fastapi import APIRouter, Request
import structlog

logger = structlog.get_logger()
router = APIRouter()


@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint"""
    
    # Check FalkorDB connection via Retriever if available
    try:
        from app.main import get_retriever
        retriever = get_retriever()
        falkordb_healthy = True if retriever else False
    except Exception as e:
        logger.error("falkordb_health_check_failed", error=str(e))
        falkordb_healthy = False
    
    status = "healthy" if falkordb_healthy else "degraded"
    
    return {
        "status": status,
        "service": "data-vent",
        "version": "0.1.0",
        "dependencies": {
            "falkordb": {
                "healthy": falkordb_healthy
            }
        }
    }
