"""
Data Vent — FalkorDB direct query utilities.

Provides low-level access to FalkorDB via the Redis protocol
for raw Cypher queries (e.g. administrative inspection, index management).

For all retrieval use-cases, prefer GraphitiService which handles
entity/relationship aware hybrid search automatically.
"""

from __future__ import annotations

from typing import Any

import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)


class FalkorDBClient:
    """
    Thin Redis-protocol client for direct GRAPH.QUERY access.

    Use this only for raw administrative Cypher queries.
    All semantic search should go through GraphitiService.
    """

    def __init__(
        self,
        host: str,
        port: int,
        graph_name: str,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._graph_name = graph_name
        self._username = username
        self._password = password
        self._client: redis.Redis | None = None

    async def connect(self) -> None:
        """Establish a connection to FalkorDB."""
        self._client = redis.Redis(
            host=self._host,
            port=self._port,
            username=self._username,
            password=self._password,
            decode_responses=True,
        )
        # Ping to verify connectivity
        await self._client.ping()
        logger.info(
            "falkordb_client_connected",
            host=self._host,
            port=self._port,
            graph=self._graph_name,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a raw Cypher query against FalkorDB via GRAPH.QUERY."""
        if self._client is None:
            raise RuntimeError("FalkorDBClient not connected")

        cmd_args: list[Any] = [self._graph_name, cypher]
        if params:
            # FalkorDB accepts PARAMS clause inline; embed directly in cypher
            pass
        result = await self._client.execute_command("GRAPH.QUERY", *cmd_args)
        return result

    async def graph_info(self) -> dict[str, Any]:
        """Return basic graph statistics."""
        if self._client is None:
            raise RuntimeError("FalkorDBClient not connected")
        raw = await self._client.execute_command(
            "GRAPH.QUERY", self._graph_name, "MATCH (n) RETURN count(n) AS node_count"
        )
        return {"raw": raw}


def build_client_from_settings(settings: Any) -> FalkorDBClient:
    """Factory: create a FalkorDBClient from app settings."""
    return FalkorDBClient(
        host=settings.FALKORDB_HOST,
        port=settings.FALKORDB_PORT,
        graph_name=settings.FALKORDB_GRAPH_NAME,
        username=settings.FALKORDB_USERNAME or None,
        password=settings.FALKORDB_PASSWORD or None,
    )
