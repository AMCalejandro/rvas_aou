FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install essential packages
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    moreutils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pixi (package manager)
RUN curl -fsSL https://pixi.sh/install.sh | sh

# Add pixi to PATH for all users
ENV PATH="/root/.pixi/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]