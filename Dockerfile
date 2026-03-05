FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY data-vent/pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir -e .

# Copy application code and proto
COPY data-vent/app ./app
COPY data-vent/proto ./proto

# Generate gRPC stubs
RUN python -m grpc_tools.protoc \
    -I proto \
    --python_out=app/proto \
    --grpc_python_out=app/proto \
    proto/retrieval.proto || true

# Expose ports (HTTP + gRPC)
EXPOSE 3005 50056

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:3005/health'); r.raise_for_status()"

# Start the service
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3005"]
