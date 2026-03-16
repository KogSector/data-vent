# Data Vent Service

## Overview

Data Vent is a data streaming and export service for the ConFuse platform. It provides real-time data access and export capabilities for various data sources including the FalkorDB knowledge graph.

## Features

- **Real-time Streaming**: Live data streaming from ConFuse services
- **Export Capabilities**: Multiple export formats (JSON, CSV, Parquet)
- **Graph Data Export**: Specialized export for FalkorDB graph data
- **Event Streaming**: Kafka-based event distribution
- **REST API**: HTTP endpoints for data access

## Architecture

```
data-vent/
├── src/
│   ├── main.rs              # Main application entry point
│   ├── streaming/           # Streaming functionality
│   ├── export/              # Export handlers
│   └── api/                 # REST API endpoints
├── Cargo.toml               # Rust dependencies
├── Dockerfile               # Container configuration
└── README.md                # Service documentation
```

## Quick Start

```bash
# Install Rust dependencies
cargo build --release

# Set environment variables
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export FALCORDB_HOST=localhost
export FALCORDB_PORT=6379

# Run the service
cargo run --bin data-vent
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/export/graph` | POST | Export graph data |
| `/stream/events` | GET | Stream events |
| `/export/entities` | POST | Export entities |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | 8080 | Server port |
| `KAFKA_BOOTSTRAP_SERVERS` | Yes | localhost:9092 | Kafka bootstrap servers |
| `FALCORDB_HOST` | Yes | localhost | FalkorDB host |
| `FALCORDB_PORT` | Yes | 6379 | FalkorDB port |
| `EXPORT_DIR` | No | ./exports | Export directory |

## Technology Stack

- **Language**: Rust
- **Web Framework**: Axum
- **Streaming**: Kafka
- **Database**: FalkorDB (for graph exports)
- **Serialization**: Serde

## Integration

- **ConFuse Platform**: Native integration
- **FalkorDB**: Graph data source
- **Kafka**: Event streaming
- **File Storage**: Export files
