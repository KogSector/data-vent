"""
Data Vent — Configuration Management
"""
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Service ────────────────────────────────────────────────────────────────
    APP_PORT: int = Field(default=3005, alias="DATA_VENT_PORT")
    HOST: str = "0.0.0.0"
    GRPC_PORT: int = 50056
    GRPC_HOST: str = "0.0.0.0"
    ENVIRONMENT: str = "production"

    # ── FalkorDB ───────────────────────────────────────────────────────────────
    FALKORDB_HOST: str = "localhost"
    FALKORDB_PORT: int = 6379
    FALKORDB_USERNAME: str = "default"
    FALKORDB_PASSWORD: str = ""
    FALKORDB_DATABASE: int = 0
    FALKORDB_GRAPH_NAME: str = "knowledge-layer"
    FALKORDB_VECTOR_DIMENSION: int = 384
    FALKORDB_SIMILARITY_THRESHOLD: float = 0.75
    FALKORDB_MAX_RESULTS: int = 100


    # ── Graphify (new backend — feature-flagged) ──────────────────────────────
    # Removed as Graphify is deprecated

    # ── Downstream Services ───────────────────────────────────────────────────
    EMBEDDINGS_GRPC_ADDR: str = "embeddings-service:50054"
    EMBEDDINGS_SERVICE_URL: str = "http://localhost:3001"
    CLIENT_CONNECTOR_URL: str = "http://localhost:3004"
    CLIENT_CONNECTOR_GRPC_ADDR: str = "client-connector:50055"
    FEATURE_TOGGLE_SERVICE_URL: str = "http://localhost:3099"

    # ── Retrieval Pipeline ────────────────────────────────────────────────────
    PIPELINE_MAX_QUERY_CHUNKS: int = 10
    PIPELINE_PER_CHUNK_TIMEOUT: float = 10.0
    PIPELINE_VECTOR_TOP_K: int = 5
    PIPELINE_DFS_DEPTH: int = 2
    PIPELINE_DFS_MIN_RELEVANCE: float = 0.3
    PIPELINE_DFS_MAX_RESULTS: int = 20
    PIPELINE_MAX_TOTAL_RESULTS: int = 50
    PIPELINE_VECTOR_WEIGHT: float = 0.6
    PIPELINE_GRAPH_WEIGHT: float = 0.3
    PIPELINE_CROSS_CHUNK_WEIGHT: float = 0.1

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ("../.env", "../.env.secret")
        case_sensitive = True
        extra = "ignore"


# Global settings instance
settings = Settings()
