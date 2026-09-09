# Unified image: FastAPI API + Celery worker in one container.
# Target: Koyeb Free Eco (1 service, 512 MB). See start.sh.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies (consolidated across api/ and worker/) — copied first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source. api/ and worker/ keep their own flat layouts;
# start.sh puts each on PYTHONPATH.
COPY api/ ./api/
COPY worker/ ./worker/
COPY start.sh ./start.sh
RUN chmod +x ./start.sh

EXPOSE 8000

CMD ["./start.sh"]
