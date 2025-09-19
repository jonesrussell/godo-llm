# Multi-stage build for local LLM inference
FROM python:3.13-slim AS builder

# Set build arguments for better caching
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies in a single layer with cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc-12 \
    g++-12 \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set working directory
WORKDIR /app

# Set compiler environment variables for llama-cpp-python
ENV CC=gcc-12
ENV CXX=g++-12

# Copy and install Python dependencies with optimizations
COPY requirements.txt .
RUN pip install --no-cache-dir --disable-pip-version-check \
    --user --no-warn-script-location \
    -r requirements.txt

# Production stage
FROM python:3.13-slim AS production

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG APP_USER=llmuser
ARG APP_UID=1000
ARG APP_GID=1000

# Create user and group first for better layer caching
RUN groupadd -g ${APP_GID} ${APP_USER} && \
    useradd -m -u ${APP_UID} -g ${APP_GID} -s /bin/bash ${APP_USER}

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libgcc-s1 \
    libstdc++6 \
    libc6-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set working directory and create necessary directories
WORKDIR /app
RUN mkdir -p /app/models && \
    chown -R ${APP_USER}:${APP_USER} /app

# Copy Python packages from builder (using --user install path)
COPY --from=builder --chown=${APP_USER}:${APP_USER} \
    /root/.local /home/${APP_USER}/.local

# Update PATH to include user-installed packages
ENV PATH=/home/${APP_USER}/.local/bin:$PATH

# Set environment variables for better Python behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/home/${APP_USER}/.local/lib/python3.11/site-packages

# Set library path environment variable and ensure proper linking
ENV LD_LIBRARY_PATH=/home/${APP_USER}/.local/lib/python3.11/site-packages/llama_cpp/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu

# Copy application files with proper ownership
# Use specific COPY commands for better caching
COPY --chown=${APP_USER}:${APP_USER} main.py .
COPY --chown=${APP_USER}:${APP_USER} test_api.py .
COPY --chown=${APP_USER}:${APP_USER} setup_*.py .
COPY --chown=${APP_USER}:${APP_USER} .env.example .
COPY --chown=${APP_USER}:${APP_USER} frontend/ ./frontend/

# Switch to non-root user
USER ${APP_USER}

# Debug: Check library availability (remove this section after confirming it works)
RUN echo "=== Library Debug Info ===" && \
    find /home/${APP_USER}/.local -name "*.so*" -type f | head -10 && \
    ldconfig -p | grep gomp && \
    echo "=== End Debug Info ==="

# Expose port
EXPOSE 8000

# Improved health check with better error handling
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use exec form and specify signal handling
CMD ["python", "-u", "main.py"]
