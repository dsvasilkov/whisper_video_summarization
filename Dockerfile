FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ffmpeg \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.3
ENV POETRY_HOME=/opt/poetry
ENV POETRY_CACHE_DIR=/opt/poetry-cache
ENV POETRY_NO_INTERACTION=1

RUN pip install --no-cache-dir poetry==${POETRY_VERSION} \
    && poetry config virtualenvs.create false

ARG POETRY_GROUPS=""

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry install \
    --no-ansi \
    --no-root \
    --with ${POETRY_GROUPS} \
    && rm -rf $POETRY_CACHE_DIR

COPY . /app/

EXPOSE 8000 8080 8501
CMD ["/bin/bash"]
