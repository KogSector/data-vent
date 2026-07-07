# syntax=docker/dockerfile:1
FROM rust:1.77-slim as builder

WORKDIR /usr/src/app

# Install protobuf compiler for tonic
RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

COPY . .

# Build for release
RUN cargo build --release

# Runtime image
FROM debian:bookworm-slim

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/app/target/release/data-vent /usr/local/bin/data-vent

EXPOSE 3002
EXPOSE 50051

CMD ["data-vent"]
