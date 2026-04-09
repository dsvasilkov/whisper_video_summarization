# Образы Docker

- **`Dockerfile`** — FastAPI API-приложение (группа зависимостей `monitoring`), без torch/CUDA.
- **`Dockerfile.worker`** — базовый образ Celery worker для очередей `asr` и `llm`.
- **`Dockerfile.worker-pyannote`** — отдельный Celery worker для очереди `pyannote` (диаризация и merge спикеров).
- **`Dockerfile.vllm-asr`** — vLLM OpenAI server для Whisper ASR с `vllm[audio]`.
- Мониторинг выполняется связкой Prometheus + Grafana (`k8s/monitoring.yaml`).

Сборка API:

```bash
docker build -t whisper-api:latest .
```

Сборка Celery worker (общий образ для `asr`/`llm`):

```bash
docker build -f Dockerfile.worker -t whisper-worker:latest .
docker build -f Dockerfile.worker -t whisper-worker-asr:latest .
```

Сборка Celery worker для pyannote:

```bash
docker build -f Dockerfile.worker-pyannote -t whisper-worker-pyannote:latest .
```

Сборка vLLM ASR (audio):

```bash
docker build -f Dockerfile.vllm-asr -t whisper-vllm-asr:latest .
```

Сборка frontend:

```bash
docker build -t whisper-frontend:latest ./frontend
```
