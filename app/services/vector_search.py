"""
Data Vent — FalkorDB direct query utilities.

Provides low-level access to FalkorDB via the Redis protocol
for raw Cypher queries (e.g. administrative inspection, index management).
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

    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
    )

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (redis.ConnectionError, redis.TimeoutError, ConnectionError, OSError)
        ),
        before_sleep=lambda retry_state: logger.warning(
            "FalkorDB connection attempt failed, retrying",
            attempt=retry_state.attempt_number,
            error=str(retry_state.outcome.exception()),
        ),
    )
    async def connect(self) -> None:
        """Establish a connection to FalkorDB."""
        logger.info(
            f"Attempting to connect to FalkorDB at {self._host}:{self._port}..."
        )
        kwargs = {
            "host": self._host,
            "port": self._port,
            "decode_responses": True,
        }
        if self._password:
            kwargs["password"] = self._password
            if self._username:
                kwargs["username"] = self._username

        self._client = redis.Redis(**kwargs)
        # Ping to verify connectivity
        await self._client.ping()  # type: ignore
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

    async def initialize_indexes(self) -> None:
        """Initialize missing vector and full-text search indexes on the graph."""
        try:
            logger.info("Initializing vector and FTS indexes on FalkorDB...")
            # Create Vector Index
            vector_cypher = "CREATE VECTOR INDEX FOR (n:Vector_Chunk) ON (n.embeddings) OPTIONS {dimension: 768, similarityFunction: 'cosine'}"
            try:
                await self.query(vector_cypher)
                logger.info("Vector index created successfully")
            except Exception as e:
                if (
                    "already exists" not in str(e).lower()
                    and "already indexed" not in str(e).lower()
                ):
                    logger.warning(
                        "Failed to create vector index (might already exist)",
                        error=str(e),
                    )

            # Create FTS Index
            fts_cypher = (
                "CALL db.idx.fulltext.createNodeIndex('Vector_Chunk', 'content')"
            )
            try:
                await self.query(fts_cypher)
                logger.info("Full-Text Search index created successfully")
            except Exception as e:
                if (
                    "already exists" not in str(e).lower()
                    and "already registered" not in str(e).lower()
                    and "already indexed" not in str(e).lower()
                ):
                    logger.warning(
                        "Failed to create FTS index (might already exist)", error=str(e)
                    )

            logger.info("Index initialization complete")
        except Exception as e:
            logger.error("Error during index initialization", error=str(e))


def build_client_from_settings(settings: Any) -> FalkorDBClient:
    """Factory: create a FalkorDBClient from app settings."""
    return FalkorDBClient(
        host=settings.FALKORDB_HOST,
        port=settings.FALKORDB_PORT,
        graph_name=settings.FALKORDB_GRAPH_NAME,
        username=settings.FALKORDB_USERNAME or None,
        password=settings.FALKORDB_PASSWORD or None,
    )
