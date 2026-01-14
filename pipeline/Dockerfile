FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY requirements.txt ./

# Install dependencies in system environment
RUN uv pip install --system --no-cache -r requirements.txt

# Copy pipeline package contents
# Assuming build context is the 'pipeline/' directory
COPY . /app/pipeline/

# Set Python path to include the app root and the mounted project root
# The project root is mounted to /app/project in docker-compose.yml
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
