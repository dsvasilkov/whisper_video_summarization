# По умолчанию — лёгкий образ FastAPI (см. также Dockerfile.worker для GPU-воркера, Dockerfile.mlflow для MLflow).
# docker build -t whisper-api:latest .
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry-cache \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ffmpeg \
    curl \
    netcat-openbsd \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && pip install --no-cache-dir poetry==${POETRY_VERSION} \
    && poetry config virtualenvs.create false

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-ansi --no-root --only main --with mlops \
    && rm -rf "$POETRY_CACHE_DIR" /tmp/poetry-cache

COPY . /app/

RUN cd /app && git init \
    && git config user.email "docker@local" \
    && git config user.name "docker" \
    && (git add -A && git commit -m "init" || true)

EXPOSE 8000
CMD ["uvicorn", "whisper_video_summarization.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
