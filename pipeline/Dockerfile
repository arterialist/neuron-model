FROM python:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    libpq-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency metadata and lockfile (build context: pipeline/)
COPY pyproject.toml uv.lock ./

# Install locked dependencies into a virtual environment (default: ./.venv)
ENV UV_LINK_MODE=copy
RUN uv venv && uv sync --frozen --no-install-project
ENV PATH="/app/.venv/bin:$PATH"

# Copy pipeline package contents
COPY . /app/pipeline/

# Set Python path to include the app root and the mounted project root
ENV PYTHONPATH=/app:/app/project

# Create directories for mounting
RUN mkdir -p /app/experiments /app/data /app/db /app/project

# Expose port
EXPOSE 8000

# Environment variables
ENV PIPELINE_OUTPUT_DIR=/app/experiments
ENV PIPELINE_DB_PATH=/app/db/pipeline.db

# Run the API server
CMD ["python", "-m", "pipeline.api.main"]
