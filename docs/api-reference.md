# data-vent API Reference

## Overview
This document outlines the HTTP API endpoints exposed by `data-vent`.

## HTTP Endpoints

### `GET /health`
Returns the health status of the service.
- **Response**: `200 OK` with service status

### `POST /api/v1/retrieve`
Standard retrieval endpoint for intelligent search.
- **Request Body**:
  ```json
  {
    "intent": "string",
    "keywords": ["string"],
    "limit": 10,
    "falkordb_graph_name": "optional string"
  }
  ```
- **Response**: JSON with search results and metadata

### `POST /api/v1/retrieve/stream`
Streaming retrieval endpoint for real-time results via Server-Sent Events.
- **Request Body**: Same as `/api/v1/retrieve`
- **Response**: SSE stream with progressive results

### `GET /`
Root endpoint for basic health monitoring.
- **Response**: `{"status": "ok"}`
