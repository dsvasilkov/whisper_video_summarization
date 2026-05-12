# GPU (cu130): хост и Docker

## Важно

- **Драйвер NVIDIA** устанавливается только **на хост** (Windows/Linux с GPU), не внутри образа.
- В контейнер попадают **библиотеки драйвера** через **NVIDIA Container Toolkit** (`nvidia-container-toolkit`), которые сопоставляются с версией драйвера на хосте.
- В образе задаётся **CUDA 13.x + cuDNN runtime** (`nvidia/cuda:…-cudnn-runtime-…`) и **PyTorch +cu130**; это согласовано с колёсами cu130.

## Linux (рекомендуется для GPU)

1. Установите [драйвер NVIDIA](https://www.nvidia.com/Download/index.aspx) (версия, поддерживающая CUDA 13.x — см. [таблицу совместимости](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).

2. Установите **NVIDIA Container Toolkit**:
   - <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>

3. Перезапустите Docker:
   ```bash
   sudo systemctl restart docker
   ```

4. Проверка:
   ```bash
   docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
   ```

5. Сборка **воркера** (GPU) и проверка PyTorch в нём:
   ```bash
   docker build -f Dockerfile.worker -t whisper-worker:latest .
   docker run --rm --gpus all whisper-worker:latest python3 -c "import torch; print(torch.cuda.is_available())"
   ```

   Лёгкий **API** без GPU: `docker build -t whisper-api:latest .` (см. `docs/docker-images.md`).

## Windows (Docker Desktop + WSL2)

- Установите драйвер NVIDIA **на Windows**.
- Включите WSL2 и Docker Desktop с интеграцией WSL2 и **GPU support** (Settings → Resources → WSL integration / GPU).
- Запуск контейнера с GPU: `docker run --gpus all ...` из WSL2.

## Kubernetes

- Установите [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html) или **NVIDIA Device Plugin** и убедитесь, что на узлах с GPU установлен драйвер.
- В манифестах GPU обычно нужен для:
  - `ray-serve` (pyannote + embeddings + faster-whisper ASR в одном поде),
  - `vllm-llm` и/или `vllm-asr` (если запущены на GPU).
- Для `whisper-worker-asr` GPU **не** нужен, если ASR идёт в Ray (только `WHISPER_SERVE_URL`). Для `whisper-api` и `whisper-worker-llm` GPU не обязателен, если LLM на отдельном vLLM-поде.

## Диаризация: флаги окружения

- в **`whisper-api`**: `PYANNOTE_ENABLED=true`, чтобы задача создавалась с диаризацией (клиент может отключить через `force_disable_diarization`);
- в **`whisper-worker-asr`**: `PYANNOTE_SERVE_URL` — базовый URL приложения pyannote в Ray Serve (ASR и диаризация выполняются параллельно в коде транскрипции);
- в **поде Ray Serve**: `PYANNOTE_HF_TOKEN` (секрет `pyannote-secrets`), модель задаётся в `k8s/ray-serve.yaml`.

Рекомендуется при необходимости держать `PYANNOTE_ENABLED` и `PYANNOTE_PIPELINE_ENABLED` согласованными для локального/in-process pyannote.

## Переменные окружения в образе

| Переменная | Назначение |
|------------|------------|
| `NVIDIA_VISIBLE_DEVICES` | Какие GPU видны (по умолчанию `all`) |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility,video` — нужно для CUDA/видео |
