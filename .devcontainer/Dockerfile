# Agent Bible Development Container
# Python 3.11 base with research code development tools

FROM mcr.microsoft.com/devcontainers/python:3.11

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for scientific computing and C++ development
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python development tools globally
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    ruff \
    mypy \
    pytest \
    pytest-cov \
    pytest-xdist \
    pre-commit \
    hypothesis \
    numpy \
    scipy

# Set up workspace
WORKDIR /workspace

# Create directory for opencode context manager
RUN mkdir -p /home/vscode/.local/share/opencode && \
    chown -R vscode:vscode /home/vscode/.local

# Switch to non-root user
USER vscode

# Set Python to not write bytecode and be unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Default command
CMD ["bash"]
