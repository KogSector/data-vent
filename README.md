# Data-Vent Service

Semantic search and graph traversal service using Graphiti and FalcorDB for the ConFuse platform.

## Overview

Data-Vent is a Python-based service that provides semantic search capabilities and knowledge graph traversal using Graphiti's temporal knowledge graph framework with FalcorDB (Neo4j) as the backend storage.

## Technology Stack

- **Framework**: FastAPI with uvicorn
- **Python Version**: 3.11+
- **Knowledge Graph**: Graphiti Core (v0.26.0+)
- **Database**: Neo4j (v5.14.0+) via FalcorDB
- **Validation**: Pydantic v2
- **HTTP Client**: httpx
- **Logging**: structlog

## Dependencies

### Core Dependencies

```toml
fastapi>=0.109.0           # Web framework
uvicorn[standard]>=0.27.0  # ASGI server
pydantic>=2.5.0            # Data validation
pydantic-settings>=2.1.0   # Configuration management
graphiti-core>=0.26.0      # Temporal knowledge graph
neo4j>=5.14.0              # Neo4j driver
httpx>=0.26.0              # Async HTTP client
python-dotenv>=1.0.0       # Environment configuration
structlog>=24.1.0          # Structured logging
```

### Development Dependencies

```toml
pytest>=7.4.0              # Testing framework
pytest-asyncio>=0.21.0     # Async test support
pytest-cov>=4.1.0          # Coverage reporting
```

## Installation

### Using pip

```bash
# Install core dependencies
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Using modern Python packaging

```bash
# Install from pyproject.toml
python -m pip install .
```

## Configuration

Configuration is managed through environment variables. Create a `.env` file in the service directory:

```env
# FalcorDB/Neo4j Configuration
FALCORDB_URI=bolt://localhost:6380
FALCORDB_USERNAME=neo4j
FALCORDB_PASSWORD=your_password
FALCORDB_DATABASE=neo4j

# Service Configuration
DATA_VENT_PORT=8081
DATA_VENT_HOST=0.0.0.0

# Logging
LOG_LEVEL=INFO
```

## Features

### Semantic Search
- Vector similarity search using FalcorDB's native vector capabilities
- Hybrid search combining vector embeddings and graph traversal
- Configurable similarity thresholds and result limits

### Knowledge Graph Operations
- Temporal knowledge graph using Graphiti
- Entity and relationship extraction
- Graph traversal and pattern matching
- Time-aware knowledge queries

### API Endpoints

The service provides RESTful API endpoints for:
- Semantic search queries
- Knowledge graph traversal
- Entity relationship queries
- Temporal knowledge retrieval

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=data_vent --cov-report=html

# Run async tests
pytest -v tests/
```

### Code Quality

```bash
# Linting
ruff check .

# Formatting
black .

# Type checking
mypy .
```

### Local Development

```bash
# Start the service
uvicorn data_vent.main:app --reload --port 8081

# Or using the installed package
python -m data_vent
```

## Integration with ConFuse Platform

Data-Vent integrates with other ConFuse services:

- **Embeddings Service**: Receives vector embeddings for semantic search
- **Unified Processor**: Provides search results for hybrid queries
- **MCP Server**: Exposes knowledge graph data to AI agents
- **API Backend**: Central orchestration and request routing

## Architecture

```
┌─────────────────────────────────────────┐
│           Data-Vent Service             │
├─────────────────────────────────────────┤
│  FastAPI Application                    │
│  ├── Semantic Search API                │
│  ├── Graph Traversal API                │
│  └── Knowledge Query API                │
├─────────────────────────────────────────┤
│  Services Module                        │
│  ├── graphiti_service.py               │
│  │   └── Graphiti integration          │
│  └── feature_toggle_client.py          │
│      └── Feature flag management        │
├─────────────────────────────────────────┤
│  Graphiti Core                          │
│  ├── Temporal Knowledge Graph           │
│  ├── Entity Management                  │
│  └── Relationship Tracking              │
├─────────────────────────────────────────┤
│  FalcorDB (Neo4j)                       │
│  ├── Vector Storage                     │
│  ├── Graph Database                     │
│  └── Temporal Indexing                  │
└─────────────────────────────────────────┘
```

### Service Module Structure

The `app/services/` module contains reusable service clients and integrations:

- **graphiti_service.py**: Graphiti temporal knowledge graph integration
- **feature_toggle_client.py**: Feature flag management client
- **__init__.py**: Services module initialization

## Performance Considerations

- **Async Operations**: All I/O operations use async/await for non-blocking execution
- **Connection Pooling**: Neo4j driver maintains connection pool for efficiency
- **Structured Logging**: structlog provides high-performance structured logging
- **Pydantic V2**: Leverages Rust-based validation for improved performance

## Deployment

### Container Deployment

```bash
# Build container
podman build -t data-vent:latest .

# Run container
podman run -d \
  --name data-vent \
  -p 8081:8081 \
  --env-file .env \
  data-vent:latest
```

### Kubernetes Deployment

See `dev-app-configs/kubernetes/deployments/data-vent.yaml` for Kubernetes manifests.

## Monitoring

The service exposes metrics and health endpoints:

- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics (if enabled)
- Structured logs via structlog for centralized logging

## Related Services

- **relation-graph**: Legacy knowledge graph service (being replaced)
- **embeddings-service**: Vector generation service
- **unified-processor**: Hybrid search orchestration
- **mcp-server**: AI agent protocol implementation

## Migration Notes

Data-Vent is part of the ChromaDB to FalcorDB migration effort, providing:
- Native Neo4j vector storage instead of ChromaDB
- Graphiti-based temporal knowledge graph
- Unified semantic search and graph traversal

## Support

For issues or questions:
1. Check the ConFuse platform documentation
2. Review the Graphiti Core documentation: https://github.com/getzep/graphiti
3. Consult the Neo4j documentation for FalcorDB specifics

## License

Part of the ConFuse Knowledge Intelligence Platform.
