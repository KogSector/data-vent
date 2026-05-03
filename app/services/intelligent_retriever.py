"""
Data Vent — Intelligent Retriever

Feature-flagged retrieval engine that routes queries to either:
- Graphify (new) — hybrid vector + graph search via HTTP API
- Graphiti (legacy) — hybrid vector + BM25 + graph via graphiti-core + FalkorDB

Feature flag: `GRAPHIFY_RETRIEVAL_ENABLED`
- When enabled: queries go to GraphifyService
- When disabled (default): queries go to GraphitiService (existing behaviour)
"""

from __future__ import annotations

import os
from typing import Any

import structlog

from app.services.graphiti_service import GraphitiService

logger = structlog.get_logger(__name__)

# Metadata keys extracted once for Graphify results — avoids
# re-creating the same tuple on every search call.
_GRAPHIFY_META_KEYS = ("id", "source_type", "source_id", "chunk_type", "content_path", "language")


class SearchResult:
    """Normalised retrieval result (backend-agnostic)."""

    __slots__ = ("content", "score", "metadata", "source")

    def __init__(self, content: str, score: float, metadata: dict[str, Any], source: str = "graphiti") -> None:
        self.content = content
        self.score = score
        self.metadata = metadata
        self.source = source

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content, "score": self.score, "metadata": self.metadata, "source": self.source}


class IntelligentRetriever:
    """
    Retrieval engine with feature-flagged backend selection.

    Supports two backends:
    - **Graphify** (new): HTTP-based hybrid search via GraphifyService
    - **Graphiti** (legacy): graphiti-core + FalkorDB

    The active backend is selected by the ``GRAPHIFY_RETRIEVAL_ENABLED`` env var.
    When the flag is disabled or the Graphify service is unavailable, queries
    automatically fall back to Graphiti.
    """

    __slots__ = ("_graphiti", "_graphify", "_graphify_enabled")

    def __init__(self, graphiti_service: GraphitiService, graphify_service: Any | None = None) -> None:
        self._graphiti = graphiti_service
        self._graphify = graphify_service
        self._graphify_enabled: bool = os.environ.get(
            "GRAPHIFY_RETRIEVAL_ENABLED", "false"
        ).lower() in ("true", "1", "yes")

    @property
    def active_backend(self) -> str:
        """Return the name of the currently active retrieval backend."""
        if self._graphify_enabled and self._graphify and self._graphify.is_available:
            return "graphify"
        return "graphiti"

    # ── Public API ────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        group_ids: list[str] | None = None,
        num_results: int = 10,
        center_node_uuid: str | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve relevant knowledge from the active backend.

        Tries Graphify first (if enabled + available), falls back to Graphiti.
        """
        backend = self.active_backend
        logger.info("retriever_search", query=query[:80], backend=backend)

        if backend == "graphify":
            try:
                return await self._retrieve_graphify(query, group_ids, num_results, center_node_uuid)
            except Exception as exc:
                logger.warning("graphify_retrieval_fallback", error=str(exc))

        return await self._retrieve_graphiti(query, group_ids, num_results, center_node_uuid)

    async def retrieve_with_context(
        self,
        query: str,
        entity_uuid: str,
        group_ids: list[str] | None = None,
        num_results: int = 15,
    ) -> list[SearchResult]:
        """Retrieve knowledge centred around a specific entity node."""
        return await self.retrieve(query, group_ids, num_results, center_node_uuid=entity_uuid)

    # ── Private: Graphify ─────────────────────────────────────────────────────

    async def _retrieve_graphify(
        self, query: str, group_ids: list[str] | None, num_results: int, center_node_uuid: str | None,
    ) -> list[SearchResult]:
        raw = await self._graphify.search(
            query=query, group_ids=group_ids, num_results=num_results, center_node_uuid=center_node_uuid,
        )
        results = [
            SearchResult(
                content=r.get("chunk_text") or r.get("content", ""),
                score=float(r.get("score", 0.0)),
                metadata={k: r.get(k) for k in _GRAPHIFY_META_KEYS},
                source="graphify",
            )
            for r in raw
        ]
        logger.info("retriever_search_done", result_count=len(results), backend="graphify")
        return results

    # ── Private: Graphiti (legacy) ────────────────────────────────────────────

    async def _retrieve_graphiti(
        self, query: str, group_ids: list[str] | None, num_results: int, center_node_uuid: str | None,
    ) -> list[SearchResult]:
        raw = await self._graphiti.search(
            query=query, group_ids=group_ids, center_node_uuid=center_node_uuid, num_results=num_results,
        )
        results = [
            SearchResult(
                content=getattr(r, "fact", "") or getattr(r, "content", ""),
                score=float(getattr(r, "score", 0.0) or 0.0),
                metadata={
                    "uuid": getattr(r, "uuid", None),
                    "name": getattr(r, "name", None),
                    "created_at": str(getattr(r, "created_at", "")),
                    "valid_at": str(getattr(r, "valid_at", "")),
                },
            )
            for r in raw
        ]
        logger.info("retriever_search_done", result_count=len(results), backend="graphiti")
        return results
