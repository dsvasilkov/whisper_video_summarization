# Whisper Video Summarization

Проект для автоматической суммаризации видео с использованием моделей Whisper и Qwen (через vLLM). Система транскрибирует видеофайлы с помощью Whisper и создает краткие резюме транскрипций через OpenAI-совместимый API vLLM.

## Описание проекта

Проект представляет собой систему суммаризации видео, которая объединяет:

- **Whisper** - модель от OpenAI для транскрипции аудио и видео в текст
- **Qwen (vLLM)** - LLM для генерации суммаризаций текста на русском языке
- **FastAPI** - REST API (producer): постановка задач в очередь, получение статусов из БД
- **AIO-Celery + RabbitMQ** - очередь задач: async consumer обрабатывает инференс в воркерах
- **PostgreSQL + SQLAlchemy** - хранение статусов и результатов задач инференса
- **React + TypeScript** - веб-интерфейс (Vite) для загрузки медиа, просмотра транскрипции/диаризации и суммаризации
- **Prometheus + Grafana** - мониторинг инференса Whisper/Qwen (GPU/RAM/CPU, RPM, tokens/sec, context length)
- **DVC** - управление версиями данных и моделей
- **Hydra** - управление конфигурациями

Модели vLLM, которые хранятся в DVC:
- `openai/whisper-large-v3` -> `whisper_video_summarization/models/vllm/openai__whisper-large-v3`
- `Qwen/Qwen3.5-9B` -> `whisper_video_summarization/models/vllm/Qwen__Qwen3.5-9B`

### Основные возможности

1. **Транскрипция видео**: Преобразование видеофайлов в текст с помощью Whisper
2. **Суммаризация текста**: Создание кратких резюме из транскрипций с помощью Qwen/vLLM
3. **Инференс через очередь**: Асинхронная обработка задач через RabbitMQ + Celery worker
4. **Мониторинг инференса**: метрики через Prometheus и дашборды в Grafana
5. **Веб-интерфейс**: Удобный интерфейс для работы с системой (React + TypeScript)

## Технические детали

### Развёртывание (Docker и Kubernetes)

Основной способ запуска в продакшене — **Kubernetes**: PostgreSQL, RabbitMQ, API, Celery worker (GPU), Prometheus, Grafana и frontend, Ingress. Docker Compose в репозитории **не используется**. Типичные манифесты лежат в каталоге **`k8s/`** (namespace `whisper`), если он есть в вашей ветке.

#### Требования

- Kubernetes-кластер (в т.ч. minikube) с Ingress Controller
- Docker (для сборки образов ниже)
- Для GPU-воркера: драйвер NVIDIA на узлах и [NVIDIA Device Plugin / GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html); см. также `docs/gpu-docker-host.md`

#### Образы: назначение и размер (ориентир)

| Тег образа | Файл | Назначение |
|------------|------|------------|
| **whisper-api** (~0.5–2 GB) | [`Dockerfile`](Dockerfile) | FastAPI: очередь, БД, DVC, **без** PyTorch/CUDA |
| **whisper-worker** (~10–12 GB) | [`Dockerfile.worker`](Dockerfile.worker) | Celery worker для очереди `llm` (суммаризация) |
| **whisper-worker-asr** (~10–12 GB) | [`Dockerfile.worker`](Dockerfile.worker) | Celery worker для очереди `asr` (ASR через vLLM) |
| **whisper-worker-pyannote** | [`Dockerfile.worker-pyannote`](Dockerfile.worker-pyannote) | Celery worker для очереди `pyannote` (диаризация и merge спикеров) |
| **whisper-vllm-asr** | [`Dockerfile.vllm-asr`](Dockerfile.vllm-asr) | vLLM OpenAI server для ASR (`vllm[audio]`) |
| **whisper-frontend** | [`frontend/Dockerfile`](frontend/Dockerfile) | статика + nginx |

Раньше один тяжёлый образ покрывал все сервисы. Сейчас роли разделены; кратко см. также [`docs/docker-images.md`](docs/docker-images.md).

#### Что внутри каждого Dockerfile

**[`Dockerfile`](Dockerfile) (API)** — база `python:3.12-slim-bookworm`, зависимости через **uv**: базовые + группа **`monitoring`** (`uv sync --group monitoring`). Старт: `uvicorn` на порту **8000**. Есть `ffmpeg` и инициализация git-репозитория в `/app` для DVC. PyTorch и CUDA **не** устанавливаются.

**[`Dockerfile.worker`](Dockerfile.worker) (инференс на GPU)** — база `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04`, переменные `NVIDIA_VISIBLE_DEVICES` / `NVIDIA_DRIVER_CAPABILITIES`. Зависимости через **uv**: базовые + группы из `UV_GROUP_FLAGS` (по умолчанию **`--group inference --group monitoring`**: PyTorch с индекса `cu130`, Whisper, transformers и метрики инференса). При сборке выполняется **`dvc repro whisper_model`** — скачивается чекпоинт Whisper; модель задаётся build-arg **`WHISPER_MODEL`** (по умолчанию `large-v3`). **По умолчанию `CMD` — `/bin/bash`** — в Kubernetes или `docker run` нужно явно указать команду Celery worker (см. ниже). Порты в образе объявлены для совместимости с отладкой; рабочая нагрузка — очередь RabbitMQ.

Для мониторинга инференса используйте манифест **`k8s/monitoring.yaml`** (Prometheus + Grafana).

#### Сборка образов

```bash
# пример: minikube — собирать в docker-демоне кластера
eval $(minikube docker-env)
docker build -t whisper-api:latest .
docker build -f Dockerfile.worker -t whisper-worker:latest .
docker build -f Dockerfile.worker -t whisper-worker-asr:latest .
docker build -f Dockerfile.worker-pyannote -t whisper-worker-pyannote:latest .
docker build -f Dockerfile.vllm-asr -t whisper-vllm-asr:latest .
docker build -t whisper-frontend:latest ./frontend
```

#### Kubernetes: типичная схема

- **Namespace:** например `whisper`.
- **Компоненты:** PostgreSQL, RabbitMQ, Deployment **API** (`whisper-api`), Celery workers (**`whisper-worker-asr`**, **`whisper-worker-llm`**, **`whisper-worker-pyannote`**), vLLM-сервисы, **Prometheus**, **Grafana**, **frontend**, **Ingress**.
- **Переменные окружения (важно):**
  - **`DATABASE_URL`** — строка SQLAlchemy для API и worker (одна и та же БД), например `postgresql+psycopg2://user:pass@postgres:5432/whisper_inference`.
  - **`CELERY_BROKER_URL`** — брокер RabbitMQ, например `amqp://guest:guest@rabbitmq.whisper.svc.cluster.local:5672//` (подставьте имя сервиса из ваших манифестов).
  - **`PYANNOTE_ENABLED`** / **`PYANNOTE_PIPELINE_ENABLED`** — включение диаризации (рекомендуется задавать оба флага одинаково в `api` и `pyannote-worker`).
  - **`PYANNOTE_HF_TOKEN`** — токен HuggingFace для pyannote pipeline (только в `pyannote-worker`).
- **Тома:** общий PVC для загрузок (например **`whisper-uploads`**) с монтированием в **`/app/data`** и у API, и у worker, чтобы путь к файлу из API совпадал с путём в воркере.
- **Worker:** в `resources.limits` укажите **`nvidia.com/gpu: 1`** (или по политике кластера). Команда контейнера должна запускать Celery, например:

```bash
aio_celery worker whisper_video_summarization.celery_app.app:celery_app --loglevel=info
```

- **Ingress:** фронтенд на корне, API за префиксом **`/api`** (как в nginx фронта).

```bash
kubectl apply -f k8s/
```

Доступ к UI настраивается через Ingress (пример хоста `whisper.local` в `k8s/ingress.yaml`, если файл есть в репозитории).

Отдельные конфиги Nginx без Ingress: каталог **`nginx/`** (если присутствует).

### API и контракт очередей

**Эндпоинты FastAPI (producer):**

- `POST /api/uploads/audio` - загрузить медиафайл и создать задачу инференса → `{ "task_id": "uuid" }`
  - параметр формы: `force_disable_diarization` (`true/false`)
- `GET /api/tasks/{task_id}` - статус и результат задачи
- `GET /api/tasks?limit=50&offset=0` - список задач с пагинацией
- `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/forgot-password`, `POST /api/auth/reset-password`

**Фоновая обработка (`aio_celery`):**

- очередь `asr`: транскрипция через vLLM ASR;
- очередь `pyannote`: подготовка диаризации и merge спикеров в транскрипцию;
- очередь `llm`: суммаризация финальной транскрипции.

Если диаризация включена, задача `llm` отправляется после merge спикеров; если выключена/пропущена — после ASR.

**Frontend -> API:**

1. Frontend конвертирует исходный файл (video/audio) в WAV через `ffmpeg.wasm`.
2. Frontend отправляет файл в `POST /api/uploads/audio`.
3. Frontend опрашивает `GET /api/tasks/{task_id}` до получения итогового результата.

**Получение статуса и результата из БД:**

- `GET /tasks/{task_id}` - статус и результат задачи (pending / processing / completed / failed)
- `GET /tasks?limit=50&offset=0` - список задач с пагинацией

### Структура проекта

```
whisper_video_summarization/
├── configs/              # Конфигурационные файлы Hydra
│   ├── model/
│   ├── logging/
│   └── paths/
├── whisper_video_summarization/
│   ├── api/              # FastAPI приложение
│   ├── llm/              # Модуль суммаризации через vLLM/OpenAI API
│   ├── models/           # Модели и артефакты
│   ├── whisper/          # Модуль транскрипции
│   ├── streamlit/        # (устарел) ранее Streamlit интерфейс
│   └── utils/            # Утилиты
├── frontend/             # React + TypeScript (Vite) интерфейс
├── Dockerfile            # образ API (FastAPI, без PyTorch)
├── Dockerfile.worker     # образ Celery + Whisper/GPU
├── Dockerfile.worker-pyannote  # образ pyannote worker
├── k8s/                  # Kubernetes: Postgres, RabbitMQ, API, worker, frontend, ingress
├── pyproject.toml        # uv/PEP 621 конфигурация
└── README.md
```

### Требования

- **Kubernetes** (или локальная разработка без кластера: uv + `npm run dev` для фронта)
- **Docker** — для сборки образов приложения

### Данные и тома

В Kubernetes загрузки API и воркера должны смотреть на один и тот же путь (часто PVC **`whisper-uploads`**, mount в **`/app/data`** у обоих Deployment). Имена манифестов в **`k8s/`** могут отличаться — сверьтесь с вашей веткой. Локально: каталоги `data/`, `configs/`, при необходимости MLflow (`mlruns/`, `mlflow.db`).

## Локальная разработка фронтенда

Без Docker, с проксированием запросов к API:

```bash
cd frontend
npm install
npm run dev
```

Примечание: frontend использует `ffmpeg.wasm`, поэтому первый запуск конвертации может занять больше времени (загрузка wasm-core).

Откройте http://localhost:5173. Запросы к `/api` будут проксироваться на http://127.0.0.1:8000 (запустите FastAPI отдельно).
