# The API and the scheduled jobs run from this one image.
FROM python:3.12-slim

# libgomp is LightGBM's threading runtime; postgresql-client gives pg_dump for backups.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 postgresql-client curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Requirements first, so code changes do not reinstall the world.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY alembic.ini pytest.ini ./
COPY alembic ./alembic
COPY app ./app
COPY scripts ./scripts

# The raw payload archive and backups live on a mounted volume, not in the image.
RUN mkdir -p data/raw data/logs data/backups

EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=10s --start-period=20s \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
