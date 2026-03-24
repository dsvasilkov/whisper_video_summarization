# Образы Docker

- **`Dockerfile`** — FastAPI (`--only main --with mlops`), без torch/CUDA.
- **`Dockerfile.worker`** — Celery + полный стек inference (PyTorch **cu130**, CUDA 13 runtime).
- **`Dockerfile.mlflow`** — только `mlflow server`, минимальный размер.

Сборка воркера:

```bash
docker build -f Dockerfile.worker -t whisper-worker:latest .
```

Сборка API:

```bash
docker build -t whisper-api:latest .
```
