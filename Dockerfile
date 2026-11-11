# ==============================================================================
# Edge LLM Quantized MLOps - Multi-Stage Production Containerfile
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Native Compiler & Wheel Builder
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install core build toolchain and BLAS development headers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Optimize CPU vectorization for high-throughput edge execution (AVX, AVX2, FMA)
ENV CMAKE_ARGS="-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DLLAMA_NATIVE=OFF"
ENV FORCE_CMAKE=1
ENV PIP_NO_CACHE_DIR=1

# Upgrade build tooling
RUN pip install --upgrade pip wheel setuptools

# Copy dependency manifest
COPY requirements.txt .

# Pre-compile native wheel packages
RUN pip wheel --wheel-dir=/build/wheels -r requirements.txt


# ------------------------------------------------------------------------------
# Stage 2: Minimal Production Edge Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runner

# Create dedicated non-root user and group
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Install runtime utilities (curl for container healthcheck probe, OpenBLAS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Install pre-compiled wheels from builder stage
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy application artifacts
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY data/sample_logs.jsonl /app/data/sample_logs.jsonl

# Create models mount directory and set ownership
RUN mkdir -p /app/models_gguf /app/checkpoints && \
    chown -R appuser:appgroup /app

# Switch to non-root execution context
USER appuser

# Expose API port
EXPOSE 8000

# Default environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH="/app/models_gguf/qwen2.5-1.5b-telemetry-q4_k_m.gguf" \
    N_THREADS=2 \
    N_CTX=1024 \
    HOST="0.0.0.0" \
    PORT=8000

# Container health monitoring probe
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production server entrypoint
CMD ["python", "-m", "src.serve", "--host", "0.0.0.0", "--port", "8000"]

