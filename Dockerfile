FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy requirements
COPY requirements.txt .

# Install dependencies using uv
RUN uv pip install --system -r requirements.txt

# Copy application code
COPY . .

# Expose server port
EXPOSE 8000

# Set python path
ENV PYTHONPATH=/app

# Default command: run the server
CMD ["uvicorn", "pipeline.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
