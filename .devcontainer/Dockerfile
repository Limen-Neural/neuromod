FROM rust:1.97.1-slim-bookworm

# OpenSSL headers are only needed when the optional `sentry` feature is enabled.
# pkg-config is used by `openssl-sys` to locate the system SSL library.
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Add components used by CI, local review, and the dev container post-create step.
RUN rustup component add rustfmt clippy llvm-tools-preview

# Create a non-root user for VS Code and grant ownership of the Rust toolchain.
RUN useradd -m -s /bin/bash -u 1000 vscode \
    && chown -R vscode:vscode /usr/local/cargo /usr/local/rustup

USER vscode
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
WORKDIR /workspace
