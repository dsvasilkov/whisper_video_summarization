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
- В манифесте воркера уже указан ресурс `nvidia.com/gpu: 1`.

## Переменные окружения в образе

| Переменная | Назначение |
|------------|------------|
| `NVIDIA_VISIBLE_DEVICES` | Какие GPU видны (по умолчанию `all`) |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility,video` — нужно для CUDA/видео |
