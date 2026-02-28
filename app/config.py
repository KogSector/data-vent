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
    
    # Kafka Streaming Pipeline
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    
    # Graphiti Configuration
    GRAPHITI_LLM_PROVIDER: str = "ollama"
    GRAPHITI_LLM_MODEL: str = "llama3.2"
    GRAPHITI_LLM_ENDPOINT: str = "http://localhost:11434"
    GRAPHITI_EMBEDDING_DIM: int = 384
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
