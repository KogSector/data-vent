"""
Data Vent — Configuration Management
"""
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Service ────────────────────────────────────────────────────────────────
    APP_PORT: int = Field(alias="DATA_VENT_PORT")
    HOST: str
    GRPC_PORT: int
    GRPC_HOST: str
    ENVIRONMENT: str

    # ── FalkorDB ───────────────────────────────────────────────────────────────
    FALKORDB_HOST: str
    FALKORDB_PORT: int
    FALKORDB_USERNAME: str
    FALKORDB_PASSWORD: Optional[str] = None
    FALKORDB_DATABASE: int
    FALKORDB_GRAPH_NAME: str
    FALKORDB_VECTOR_DIMENSION: int
    FALKORDB_SIMILARITY_THRESHOLD: float
    FALKORDB_MAX_RESULTS: int


    # ── Graphify (new backend — feature-flagged) ──────────────────────────────
    # Removed as Graphify is deprecated

    # ── Downstream Services ───────────────────────────────────────────────────
    EMBEDDINGS_GRPC_ADDR: str
    EMBEDDINGS_SERVICE_URL: str
    CLIENT_CONNECTOR_URL: str
    CLIENT_CONNECTOR_GRPC_ADDR: str
    FEATURE_TOGGLE_SERVICE_URL: str

    # ── Retrieval Pipeline ────────────────────────────────────────────────────
    PIPELINE_MAX_QUERY_CHUNKS: int
    PIPELINE_PER_CHUNK_TIMEOUT: float
    PIPELINE_VECTOR_TOP_K: int
    PIPELINE_DFS_DEPTH: int
    PIPELINE_DFS_MIN_RELEVANCE: float
    PIPELINE_DFS_MAX_RESULTS: int
    PIPELINE_MAX_TOTAL_RESULTS: int
    PIPELINE_VECTOR_WEIGHT: float
    PIPELINE_GRAPH_WEIGHT: float
    PIPELINE_CROSS_CHUNK_WEIGHT: float

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str

    class Config:
        env_file = (".env.map", ".env.secret")
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()
