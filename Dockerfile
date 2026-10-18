FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as non-root — the default Docker user is root, which is an
# unnecessary privilege escalation if the container is ever compromised.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 7860

# Lets `docker ps` / compose / k8s see container health without an
# external prober. Hits the liveness endpoint, not /ready, since a
# container is "healthy" (should not be restarted) even while its DB
# dependency is briefly down — /ready governs traffic routing, not
# container lifecycle.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
