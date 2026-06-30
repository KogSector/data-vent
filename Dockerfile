# =============================================================================
# Data Vent Service - Dockerfile
# Port: 3005 (HTTP), 50056 (gRPC)
# Role: Intelligent retrieval and search
# =============================================================================

# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies for building packages and stubs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy application and proto files
COPY app ./app
COPY proto ./proto

# Generate gRPC stubs inside the app/proto directory
RUN python -m grpc_tools.protoc \
    -I proto \
    --python_out=app/proto \
    --grpc_python_out=app/proto \
    proto/retrieval.proto || true

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code (with generated stubs) and proto with strict ownership
COPY --chown=appuser:appgroup --from=builder /app/app ./app
COPY --chown=appuser:appgroup --from=builder /app/proto ./proto

# Switch to non-root user
USER appuser

# Environment explicitly defined for robustness
ENV PORT=3005
ENV PYTHONPATH=/app

# Expose ports (HTTP + gRPC)
EXPOSE 3005 50056

# Health check optimized for Render (using curl instead of python httpx)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Use dumb-init as PID 1 for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

# Start the service
CMD sh -c "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"
