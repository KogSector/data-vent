"""
Data Vent — Graphiti Knowledge Graph Service

Central service wrapping graphiti-core with FalkorDB driver.
FalkorDB connection is fully parameterised via environment variables so
switching between local Podman and any FalkorDB SaaS requires only
changing FALKORDB_HOST, FALKORDB_PORT, and FALKORDB_PASSWORD in .env.secret.
"""

import logging
from typing import Any, List
import structlog

logger = structlog.get_logger(__name__)

class GraphitiService:
    def __init__(self, config: Any = None):
        self.config = config
        self.graphiti = None
    
    async def initialize(self):
        logger.info("Mock GraphitiService initialized")
    
    async def close(self):
        pass
    
    async def search(self, *args, **kwargs):
        return []
