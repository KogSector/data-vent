# =============================================================================
# Data Vent Service - Dockerfile
# Port: 3005 (HTTP), 50056 (gRPC)
# Role: Intelligent retrieval and search
# =============================================================================

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir -e .

# Copy application code and proto with strict ownership
COPY --chown=appuser:appgroup app ./app
COPY --chown=appuser:appgroup proto ./proto

# Generate gRPC stubs
RUN python -m grpc_tools.protoc \
    -I proto \
    --python_out=app/proto \
    --grpc_python_out=app/proto \
    proto/retrieval.proto || true

# Change ownership of the workspace
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Environment explicitly defined for robustness
ENV PORT=3005

# Expose ports (HTTP + gRPC)
EXPOSE 3005 50056

# Health check optimized for Render (using curl instead of python httpx)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Use dumb-init as PID 1 for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

# Start the service
CMD sh -c "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"
