FROM rust:1.97.1-slim-bookworm

# Add components used by CI, local review, and the dev container post-create step.
RUN rustup component add rustfmt clippy llvm-tools-preview

# Create a non-root user for VS Code and grant ownership of the Rust toolchain.
RUN useradd -m -s /bin/bash -u 1000 vscode \
    && chown -R vscode:vscode /usr/local/cargo /usr/local/rustup

USER vscode
ENV CARGO_HOME=/usr/local/cargo
ENV RUSTUP_HOME=/usr/local/rustup
WORKDIR /workspace
