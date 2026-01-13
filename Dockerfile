# --- Builder Stage ---
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv and Python dependencies
RUN pip install uv
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# --- Final Stage ---
FROM python:3.12-slim

WORKDIR /app

# Copy only necessary components from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application code
COPY . .

# Expose port and set runtime environment
EXPOSE 8000
ENV PYTHONPATH=/app

CMD ["uvicorn", "pipeline.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
