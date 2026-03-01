"""
Data Vent - Configuration Management
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Service Configuration
    PORT: int = 3005
    HOST: str = "0.0.0.0"
    GRPC_PORT: int = 50056
    GRPC_HOST: str = "0.0.0.0"
    ENVIRONMENT: str = "production"
    
    # FalcorDB Configuration
    FALCORDB_URI: str
    FALCORDB_USERNAME: str = "neo4j"
    FALCORDB_PASSWORD: str
    FALCORDB_DATABASE: str = "neo4j"
    FALCORDB_VECTOR_DIMENSION: int = 384
    FALCORDB_SIMILARITY_THRESHOLD: float = 0.75
    FALCORDB_MAX_RESULTS: int = 100
    
    # Downstream Service URLs (gRPC)
    EMBEDDINGS_GRPC_ADDR: str = "embeddings-service:50054"
    
    # Service URLs (HTTP)
    EMBEDDINGS_SERVICE_URL: str = "http://localhost:3001"
    CLIENT_CONNECTOR_URL: str = "http://localhost:3004"
    FEATURE_TOGGLE_SERVICE_URL: str = "http://localhost:3099"
    
    # Graphiti Configuration
    GRAPHITI_LLM_PROVIDER: str = "ollama"
    GRAPHITI_LLM_MODEL: str = "llama3.2"
    GRAPHITI_LLM_ENDPOINT: str = "http://localhost:11434"
    GRAPHITI_EMBEDDING_DIM: int = 384
    
    # Retrieval Pipeline Configuration
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
    
    # Client-Connector (upstream)
    CLIENT_CONNECTOR_GRPC_ADDR: str = "client-connector:50055"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
