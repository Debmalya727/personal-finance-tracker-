# Dockerfile — Optimized for free cloud deployments (Render, Fly.io, Railway)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system-level dependencies (minimal set — no OpenCV, no torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory (used as fallback, primary storage is Cloudinary)
RUN mkdir -p uploads

# Set environment variables for production
ENV FLASK_ENV=production
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Expose port (configurable via PORT env var, defaults to 8080)
EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

# Run with Gunicorn — 2 workers, 120s timeout
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 2 --timeout 120 --worker-tmp-dir /dev/shm app:app