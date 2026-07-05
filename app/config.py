"""
Data Vent — Configuration Management
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Service ────────────────────────────────────────────────────────────────
    APP_PORT: int = Field(alias="DATA_VENT_PORT", default=3002)
    HOST: str = Field(default="0.0.0.0")
    GRPC_PORT: int = Field(default=50051)
    GRPC_HOST: str = Field(default="0.0.0.0")
    ENVIRONMENT: str = Field(default="production")

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
    # LLM Settings
    GEMINI_API_KEY: str = Field()
    GEMINI_BASE_URL: str = Field(default="https://generativelanguage.googleapis.com")
    GEMINI_EMBEDDING_MODEL: str = Field()
    CLIENT_CONNECTOR_URL: str
    CLIENT_CONNECTOR_GRPC_ADDR: str

    # ── Retrieval Pipeline ────────────────────────────────────────────────────
    PIPELINE_MAX_QUERY_CHUNKS: int = Field(default=5)
    PIPELINE_PER_CHUNK_TIMEOUT: float = Field(default=5.0)
    PIPELINE_VECTOR_TOP_K: int = Field(default=10)
    PIPELINE_DFS_DEPTH: int = Field(default=2)
    PIPELINE_DFS_MIN_RELEVANCE: float = Field(default=0.5)
    PIPELINE_DFS_MAX_RESULTS: int = Field(default=20)
    PIPELINE_MAX_TOTAL_RESULTS: int = Field(default=50)
    PIPELINE_VECTOR_WEIGHT: float = Field(default=0.7)
    PIPELINE_GRAPH_WEIGHT: float = Field(default=0.3)
    PIPELINE_CROSS_CHUNK_WEIGHT: float = Field(default=0.1)

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO")

    class Config:
        env_file = (".env.map", ".env.secret")
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()
