# По умолчанию — лёгкий образ FastAPI (см. также Dockerfile.worker для GPU-воркера).
# docker build -t whisper-api:latest .
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    curl \
    netcat-openbsd \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv

ARG UV_GROUP_FLAGS="--group inference --group monitoring"

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-install-project ${UV_GROUP_FLAGS}

COPY . /app/

RUN uv sync --frozen ${UV_GROUP_FLAGS}

EXPOSE 8000
CMD ["uvicorn", "whisper_video_summarization.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
