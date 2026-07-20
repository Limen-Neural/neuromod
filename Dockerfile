# Multi-stage Docker for neuromod (library + examples/tests)
# Builder — Rust 1.97.1 (bookworm) keeps glibc in sync with the debian:bookworm-slim runtime.
FROM rust:1.97.1-slim-bookworm AS builder
WORKDIR /app
COPY . .
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN cargo build --release --examples && \
    mkdir -p /out && \
    find target/release/examples/ -maxdepth 1 -type f -executable -exec cp {} /out/ \;

# Runtime example (minimal)
FROM debian:bookworm-slim
RUN useradd --system --create-home --shell /usr/sbin/nologin neuromod
WORKDIR /app
COPY --from=builder /out/ /usr/local/bin/
USER neuromod
# For library usage, typically users depend on the crate, not the image.
# This image is useful for CI reproducibility and example runs.
CMD ["ls", "/usr/local/bin"]
