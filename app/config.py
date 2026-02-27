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
    ENVIRONMENT: str = "production"
    
    # FalcorDB Configuration
    FALCORDB_URI: str
    FALCORDB_USERNAME: str = "neo4j"
    FALCORDB_PASSWORD: str
    FALCORDB_DATABASE: str = "neo4j"
    
    # Service URLs
    EMBEDDINGS_SERVICE_URL: str = "http://localhost:3001"
    CLIENT_CONNECTOR_URL: str = "http://localhost:3004"
    FEATURE_TOGGLE_SERVICE_URL: str = "http://localhost:3099"
    
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
