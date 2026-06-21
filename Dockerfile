# Multi-stage Docker for neuromod (library + examples/tests)
# Builder
FROM rust:1.80-slim AS builder
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
RUN cargo build --release --examples

# Runtime example (minimal)
FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/target/release/examples/* /usr/local/bin/
# For library usage, typically users depend on the crate, not the image.
# This image is useful for CI reproducibility and example runs.
CMD ["bash"]
