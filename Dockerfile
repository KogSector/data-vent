# ==============================================================================
# Data Vent Service - Dockerfile
# ==============================================================================
# Multi-stage build for Rust retrieval engine
# Port: 3002 (HTTP), 50051 (gRPC)
# ==============================================================================

# Stage 1: Rust builder
FROM debian:bookworm-slim AS rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    pkg-config \
    libssl-dev \
    build-essential \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup to guarantee latest stable compiler
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# ---------------------------------------------------------------------------
# Dependency caching layer
# ---------------------------------------------------------------------------
COPY Cargo.toml Cargo.lock* ./
COPY build.rs ./
COPY proto/ ./proto/

# Create dummy src for dependency caching
RUN mkdir -p src/services && \
    echo 'fn main() {}' > src/main.rs && \
    touch src/config.rs src/grpc_server.rs src/services/mod.rs

# Build dependencies (cached)
RUN cargo build --release 2>/dev/null || true

# ---------------------------------------------------------------------------
# Real build
# ---------------------------------------------------------------------------
RUN rm -rf src/*
COPY src/ ./src/

# Force Cargo to rebuild the actual source
RUN find src -type f -exec touch {} +
RUN cargo build --release

# ==============================================================================
# Stage 2: Runtime image
# ==============================================================================
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    dumb-init \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
WORKDIR /app
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app appuser

# Copy the compiled Rust binary
COPY --from=rust-builder --chown=appuser:appgroup /app/target/release/data-vent /usr/local/bin/data-vent

# Ensure correct permissions
RUN chown -R appuser:appgroup /app

ENV PORT=3002

# Switch to non-root user
USER appuser

# Health check optimized for Render
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-3002}/health || exit 1

EXPOSE 3002 50051

# Use dumb-init as PID 1 for proper signal handling
ENTRYPOINT ["dumb-init", "--"]

CMD ["data-vent"]
