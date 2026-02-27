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
    
    # Check FalcorDB connection
    try:
        graphiti_service = request.app.state.graphiti_service
        graphiti_service.driver.verify_connectivity()
        falcordb_healthy = True
    except Exception as e:
        logger.error("falcordb_health_check_failed", error=str(e))
        falcordb_healthy = False
    
    # Check feature toggle service
    try:
        feature_toggle_client = request.app.state.feature_toggle_client
        llm_enabled = await feature_toggle_client.is_enabled("enableLLM")
        toggle_healthy = True
    except Exception as e:
        logger.error("toggle_health_check_failed", error=str(e))
        toggle_healthy = False
        llm_enabled = None
    
    status = "healthy" if (falcordb_healthy and toggle_healthy) else "degraded"
    
    return {
        "status": status,
        "service": "data-vent",
        "version": "0.1.0",
        "dependencies": {
            "falcordb": {
                "healthy": falcordb_healthy
            },
            "feature_toggle": {
                "healthy": toggle_healthy,
                "llm_enabled": llm_enabled
            }
        }
    }
