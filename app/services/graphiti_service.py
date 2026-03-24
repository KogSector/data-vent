"""
Data Vent — Graphiti Knowledge Graph Service

Central service wrapping graphiti-core with FalkorDB driver.
FalkorDB connection is fully parameterised via environment variables so
switching between local Podman and any FalkorDB SaaS requires only
changing FALKORDB_HOST, FALKORDB_PORT, and FALKORDB_PASSWORD in .env.secret.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import structlog
from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.nodes import EpisodeType

logger = structlog.get_logger(__name__)


class GraphitiService:
    """
    Manages a single Graphiti instance connected to FalkorDB.

    All entity and relationship extraction is handled automatically by
    Graphiti via the configured LLM (Ollama).
    """

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._graphiti: Graphiti | None = None

    async def initialize(self) -> None:
        """Connect to FalkorDB and build Graphiti indices."""
        driver = FalkorDriver(
            host=self._settings.FALKORDB_HOST,
            port=self._settings.FALKORDB_PORT,
            username=self._settings.FALKORDB_USERNAME or None,
            password=self._settings.FALKORDB_PASSWORD or None,
        )

        self._graphiti = Graphiti(graph_driver=driver)
        await self._graphiti.build_indices_and_constraints()

        logger.info(
            "graphiti_service_ready",
            host=self._settings.FALKORDB_HOST,
            port=self._settings.FALKORDB_PORT,
            graph=self._settings.FALKORDB_GRAPH_NAME,
        )

    async def ingest(
        self,
        name: str,
        body: str,
        episode_type: EpisodeType = EpisodeType.text,
        source_description: str = "data-vent",
        group_id: str | None = None,
        reference_time: datetime | None = None,
    ) -> None:
        """Ingest an episode into Graphiti (triggers LLM entity extraction)."""
        if self._graphiti is None:
            raise RuntimeError("GraphitiService not initialised — call initialize() first")

        await self._graphiti.add_episode(
            name=name,
            episode_body=body,
            source=episode_type,
            source_description=source_description,
            reference_time=reference_time or datetime.now(timezone.utc),
            group_id=group_id or self._settings.GRAPHITI_GROUP_ID,
        )

    async def search(
        self,
        query: str,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        num_results: int = 10,
    ) -> list[Any]:
        """
        Search the Graphiti knowledge graph using hybrid vector + BM25 retrieval.

        Optionally rerank results around a known node (`center_node_uuid`)
        for graph-aware context propagation.
        """
        if self._graphiti is None:
            raise RuntimeError("GraphitiService not initialised — call initialize() first")

        kwargs: dict[str, Any] = {"num_results": num_results}
        if group_ids:
            kwargs["group_ids"] = group_ids
        if center_node_uuid:
            kwargs["center_node_uuid"] = center_node_uuid

        return await self._graphiti.search(query, **kwargs)

    async def close(self) -> None:
        """Release connections."""
        if self._graphiti is not None:
            # graphiti-core driver handles cleanup internally
            self._graphiti = None
            logger.info("graphiti_service_closed")

    @property
    def graphiti(self) -> Graphiti:
        if self._graphiti is None:
            raise RuntimeError("GraphitiService not initialised")
        return self._graphiti
