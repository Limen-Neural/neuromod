# Multi-stage Docker for neuromod (library + examples/tests)
# Builder
FROM rust:1.85-slim AS builder
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config=1.8.1-1
RUN cargo build --release --examples && \
    mkdir -p /out && \
    find target/release/examples/ -maxdepth 1 -type f -executable -exec cp {} /out/ \;

# Runtime example (minimal)
FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /out/ /usr/local/bin/
# For library usage, typically users depend on the crate, not the image.
# This image is useful for CI reproducibility and example runs.
CMD ["bash"]
