"""
Data Vent — Graphify Knowledge Graph Service

Adapter service for the Graphify knowledge graph API.
Provides episode ingestion and hybrid search (vector + graph) capabilities.

This service runs alongside GraphitiService during the migration period.
Feature flag `graphifyRetrievalEnabled` controls which backend handles queries.

Graphify API contract:
  - POST /api/v1/episodes         → ingest episode(s)
  - POST /api/v1/episodes/batch   → batch ingest
  - POST /api/v1/search/hybrid    → hybrid vector + graph search
  - GET  /health                  → health check
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from confuse_common.events.episode import GraphifyEpisode

logger = structlog.get_logger(__name__)


class GraphifyService:
    """
    Client adapter for the Graphify knowledge graph API.

    Handles:
    - Episode ingestion (single + batch)
    - Hybrid search queries (vector + graph)
    - Health monitoring

    The service URL is configured via GRAPHIFY_SERVICE_URL environment variable.
    """

    __slots__ = ("_base_url", "_timeout", "_client", "_initialized")

    def __init__(self, settings: Any) -> None:
        self._base_url: str = getattr(
            settings, "GRAPHIFY_SERVICE_URL", "http://localhost:8100"
        ).rstrip("/")
        self._timeout: float = getattr(settings, "GRAPHIFY_TIMEOUT", 30.0)
        self._client: httpx.AsyncClient | None = None
        self._initialized: bool = False

    async def initialize(self) -> None:
        """
        Initialize the HTTP client and verify Graphify connectivity.

        Called during application startup.  If Graphify is unreachable the
        service logs a warning but does not crash — the feature flag controls
        whether queries are actually routed here.
        """
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "confuse-data-vent/1.0",
            },
        )

        try:
            resp = await self._client.get("/health")
            self._initialized = resp.status_code == 200
            if self._initialized:
                logger.info("graphify_service_ready", url=self._base_url)
            else:
                logger.warning("graphify_service_unhealthy", url=self._base_url, status_code=resp.status_code)
        except Exception as exc:
            logger.warning("graphify_service_unreachable", url=self._base_url, error=str(exc))

    @property
    def is_available(self) -> bool:
        """Check if the Graphify service is initialized and reachable."""
        return self._initialized and self._client is not None

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def ingest_episode(self, episode: GraphifyEpisode) -> dict:
        """Ingest a single episode into the Graphify knowledge graph."""
        self._assert_initialized()
        try:
            resp = await self._client.post("/api/v1/episodes", json=episode.to_kafka_payload())
            resp.raise_for_status()
            logger.info("graphify_episode_ingested", episode_id=episode.id, source_type=episode.source_type.value)
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error("graphify_ingest_http_error", episode_id=episode.id, status_code=exc.response.status_code)
            return {"status": "error", "error": str(exc)}
        except Exception as exc:
            logger.error("graphify_ingest_failed", episode_id=episode.id, error=str(exc))
            return {"status": "error", "error": str(exc)}

    async def ingest_batch(self, episodes: list[GraphifyEpisode]) -> dict:
        """Batch ingest episodes into Graphify."""
        self._assert_initialized()
        try:
            payload = [ep.to_kafka_payload() for ep in episodes]
            resp = await self._client.post("/api/v1/episodes/batch", json={"episodes": payload})
            resp.raise_for_status()
            result = resp.json()
            logger.info("graphify_batch_ingested", count=len(episodes), succeeded=result.get("succeeded", 0))
            return result
        except Exception as exc:
            logger.error("graphify_batch_ingest_failed", count=len(episodes), error=str(exc))
            return {"status": "error", "total": len(episodes), "succeeded": 0, "failed": len(episodes), "error": str(exc)}

    # ── Search / Retrieval ────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        group_ids: list[str] | None = None,
        num_results: int = 10,
        center_node_uuid: str | None = None,
        source_types: list[str] | None = None,
    ) -> list[dict]:
        """Hybrid search against the Graphify knowledge graph."""
        self._assert_initialized()
        payload: dict[str, Any] = {"query": query, "num_results": num_results}
        if group_ids:
            payload["group_ids"] = group_ids
        if center_node_uuid:
            payload["center_node_uuid"] = center_node_uuid
        if source_types:
            payload["source_types"] = source_types

        try:
            resp = await self._client.post("/api/v1/search/hybrid", json=payload)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            logger.info("graphify_search_completed", query=query[:80], result_count=len(results))
            return results
        except Exception as exc:
            logger.error("graphify_search_failed", query=query[:80], error=str(exc))
            return []

    # ── Health ────────────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """Check Graphify service health."""
        if not self._client:
            return {"status": "not_initialized", "service": "graphify"}
        try:
            resp = await self._client.get("/health")
            return {"status": "healthy" if resp.status_code == 200 else "unhealthy", "service": "graphify"}
        except Exception as exc:
            return {"status": "unreachable", "service": "graphify", "error": str(exc)}

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Release HTTP client resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._initialized = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _assert_initialized(self) -> None:
        if not self._client:
            raise RuntimeError("GraphifyService not initialised — call initialize() first")
