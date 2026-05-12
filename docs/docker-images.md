# Образы Docker

- **`Dockerfile`** — FastAPI API-приложение: по умолчанию **uv** с `--group inference --group monitoring` (см. `UV_GROUP_FLAGS` в файле).
- **`Dockerfile.worker`** — базовый образ Celery worker для очередей `asr`, `llm` и `rag` (ASR по HTTP к Ray Serve `WHISPER_SERVE_URL`; диаризация — `PYANNOTE_SERVE_URL`).
- **`Dockerfile.ray`** — Ray Serve multi-app: pyannote, embeddings, faster-whisper (`Systran/faster-whisper-large-v3`, int8).
- **`Dockerfile.vllm`** — образ vLLM OpenAI server для LLM: `vllm[audio]`.
- Мониторинг выполняется связкой Prometheus + Grafana (`k8s/monitoring.yaml`).

Сборка API:

```bash
docker build -t whisper-api:latest .
```

Сборка Celery worker (общий образ для `asr` / `llm` / `rag`):

```bash
docker build -f Dockerfile.worker -t whisper-worker:latest .
docker build -f Dockerfile.worker -t whisper-worker-asr:latest .
```

Сборка Ray Serve:

```bash
docker build -f Dockerfile.ray -t whisper-ray:latest .
```

Сборка vLLM (OpenAI server):

```bash
docker build -f Dockerfile.vllm -t whisper-vllm:latest .
```

Сборка frontend:

```bash
docker build -t whisper-frontend:latest ./frontend
```
