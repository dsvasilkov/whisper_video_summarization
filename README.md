# Whisper Video Summarization

Проект для автоматической суммаризации видео с использованием моделей Whisper и T5. Система транскрибирует видеофайлы с помощью Whisper и создает краткие резюме транскрипций с помощью модели T5.

## Описание проекта

Проект представляет собой MLOps-решение для суммаризации видео, которое объединяет:

- **Whisper** - модель от OpenAI для транскрипции аудио и видео в текст
- **T5** - модель для генерации суммаризаций текста на русском языке
- **FastAPI** - REST API (producer): постановка задач в очередь, получение статусов из БД
- **Celery + RabbitMQ** - очередь задач: consumer обрабатывает инференс в воркерах
- **PostgreSQL + SQLAlchemy** - хранение статусов и результатов задач инференса
- **React + TypeScript** - веб-интерфейс (Vite) для загрузки видео и обучения модели
- **MLflow** - отслеживание экспериментов и метрик обучения
- **DVC** - управление версиями данных и моделей
- **Hydra** - управление конфигурациями

### Основные возможности

1. **Транскрипция видео**: Преобразование видеофайлов в текст с помощью Whisper
2. **Суммаризация текста**: Создание кратких резюме из транскрипций с помощью T5
3. **Обучение модели**: Fine-tuning модели T5 на датасете Gazeta
4. **Отслеживание экспериментов**: Логирование метрик и параметров через MLflow
5. **Веб-интерфейс**: Удобный интерфейс для работы с системой (React + TypeScript)

## Технические детали

### Развёртывание (Docker и Kubernetes)

Основной способ запуска в продакшене — **Kubernetes**: PostgreSQL, RabbitMQ, API, Celery worker (GPU), опционально MLflow и frontend, Ingress. Docker Compose в репозитории **не используется**. Типичные манифесты лежат в каталоге **`k8s/`** (namespace `whisper`), если он есть в вашей ветке.

#### Требования

- Kubernetes-кластер (в т.ч. minikube) с Ingress Controller
- Docker (для сборки образов ниже)
- Для GPU-воркера: драйвер NVIDIA на узлах и [NVIDIA Device Plugin / GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html); см. также `docs/gpu-docker-host.md`

#### Образы: назначение и размер (ориентир)

| Тег образа | Файл | Назначение |
|------------|------|------------|
| **whisper-api** (~0.5–2 GB) | [`Dockerfile`](Dockerfile) | FastAPI: очередь, БД, DVC, **без** PyTorch/CUDA |
| **whisper-worker** (~10–12 GB) | [`Dockerfile.worker`](Dockerfile.worker) | Celery + Whisper + PyTorch **cu130** (CUDA 13 runtime) |
| **whisper-mlflow** (~0.3–0.7 GB) | [`Dockerfile.mlflow`](Dockerfile.mlflow) | только UI MLflow (sqlite + артефакты в томе) |
| **whisper-frontend** | [`frontend/Dockerfile`](frontend/Dockerfile) | статика + nginx |

Раньше один тяжёлый образ покрывал все сервисы (~12 GB даже для API/MLflow). Сейчас роли разделены; кратко см. также [`docs/docker-images.md`](docs/docker-images.md).

#### Что внутри каждого Dockerfile

**[`Dockerfile`](Dockerfile) (API)** — база `python:3.12-slim-bookworm`, зависимости Poetry: **`main` + `mlops`** (`poetry install --only main --with mlops`). Старт: `uvicorn` на порту **8000**. Есть `ffmpeg` и инициализация git-репозитория в `/app` для DVC. PyTorch и CUDA **не** устанавливаются.

**[`Dockerfile.worker`](Dockerfile.worker) (инференс на GPU)** — база `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04`, переменные `NVIDIA_VISIBLE_DEVICES` / `NVIDIA_DRIVER_CAPABILITIES`. Poetry: **`main` + группы из `POETRY_GROUPS`** (по умолчанию **`inference,mlops`**: PyTorch с индекса `cu130`, Whisper, transformers, метрики и т.д.). При сборке выполняется **`dvc repro whisper_model`** — скачивается чекпоинт Whisper; модель задаётся build-arg **`WHISPER_MODEL`** (по умолчанию `large-v3`). **По умолчанию `CMD` — `/bin/bash`** — в Kubernetes или `docker run` нужно явно указать команду Celery worker (см. ниже). Порты в образе объявлены для совместимости с отладкой; рабочая нагрузка — очередь RabbitMQ.

**[`Dockerfile.mlflow`](Dockerfile.mlflow)** — минимальный образ: только `mlflow` из PyPI, без PyTorch. Команда: **`mlflow server`** на порту **8080**, backend **sqlite** `sqlite:////app/mlflow.db`, артефакты в **`/app/mlruns`**. Для персистентности в k8s смонтируйте PVC на `/app` или отдельно на `mlflow.db` и `mlruns`.

#### Сборка образов

```bash
# пример: minikube — собирать в docker-демоне кластера
eval $(minikube docker-env)
docker build -t whisper-api:latest .
docker build -f Dockerfile.worker -t whisper-worker:latest .
docker build -f Dockerfile.mlflow -t whisper-mlflow:latest .
docker build -t whisper-frontend:latest ./frontend
```

#### Kubernetes: типичная схема

- **Namespace:** например `whisper`.
- **Компоненты:** PostgreSQL, RabbitMQ, Deployment **API** (`whisper-api`), Deployment **Celery worker** (`whisper-worker`) с запросом **GPU**, опционально **MLflow** (`whisper-mlflow`) и **frontend**, **Ingress**.
- **Переменные окружения (важно):**
  - **`DATABASE_URL`** — строка SQLAlchemy для API и worker (одна и та же БД), например `postgresql+psycopg2://user:pass@postgres:5432/whisper_inference`.
  - **`CELERY_BROKER_URL`** — брокер RabbitMQ, например `amqp://guest:guest@rabbitmq.whisper.svc.cluster.local:5672//` (подставьте имя сервиса из ваших манифестов).
- **Тома:** общий PVC для загрузок (например **`whisper-uploads`**) с монтированием в **`/app/data`** и у API, и у worker, чтобы путь к файлу из API совпадал с путём в воркере.
- **Worker:** в `resources.limits` укажите **`nvidia.com/gpu: 1`** (или по политике кластера). Команда контейнера должна запускать Celery, например:

```bash
celery -A whisper_video_summarization.celery_app.app:celery_app worker --loglevel=info
```

- **Ingress:** фронтенд на корне, API за префиксом **`/api`** (как в nginx фронта).

```bash
kubectl apply -f k8s/
```

**Обучение через `POST /train` в лёгком API:** в образе API нет PyTorch — фоновое обучение из этого контейнера не выполнится. Обучайте с хоста (`poetry install --with inference,mlflow`) или вынесите задачу в отдельный Job/воркер с полным стеком (`Dockerfile.worker` или отдельный train-образ).

Доступ к UI настраивается через Ingress (пример хоста `whisper.local` в `k8s/ingress.yaml`, если файл есть в репозитории).

Отдельные конфиги Nginx без Ingress: каталог **`nginx/`** (если присутствует).

### Train

Для обучения модели суммаризации выполните следующие шаги:

#### 1. Подготовка данных

В проекте уже есть тестовый датасет для обучения: `data/test_train.jsonl`. Датасет содержит данные в формате JSONL, где каждая строка представляет собой JSON объект с полями `text` (текст статьи) и `summary` (краткое резюме).

Пример структуры:

```json
{"text": "Длинный текст статьи...", "summary": "Краткое резюме..."}
{"text": "Другой текст...", "summary": "Другое резюме..."}
```

#### 2. Запуск обучения

**Через веб-интерфейс (React):** откройте URL фронта (локально `npm run dev` → http://localhost:5173, в кластере — хост из Ingress) → «Обучение», загрузите датасет и нажмите «Запустить обучение».

**Ожидается успешный запуск обучения и снижение loss в процессе обучения.**

#### 3. Параметры обучения

Параметры обучения настраиваются через конфигурационные файлы в директории `configs/`:

- `configs/train.yaml` - основные параметры обучения (batch_size, epochs, seed)
- `configs/model/t5.yaml` - параметры модели (название модели, learning rate)
- `configs/data/gazeta.yaml` - параметры данных (max_length)
- `configs/paths/paths.yaml` - пути к данным и моделям

#### 4. Результаты обучения

После завершения обучения модель сохраняется в:

```
whisper_video_summarization/models/summarizer/checkpoints/best.ckpt
```

Эта модель используется для инференса. В Kubernetes модели и данные хранятся в томах/PVC или в образе — настройте монтирование под ваш сценарий.

### Использование обученной модели

**Эндпоинты FastAPI (producer — не ждёт результата, возвращает `task_id`):**

- `POST /infer` - поставить в очередь суммаризацию текста → `{ "task_id": "uuid" }`
- `POST /infer/video` - поставить в очередь транскрипцию и суммаризацию по пути к файлу → `{ "task_id": "uuid" }`
- `POST /infer/video/upload` - загрузить файл и поставить задачу в очередь → `{ "task_id": "uuid" }`
- `POST /upload/dataset` - загрузка датасета для обучения
- `POST /train` - запуск обучения (фоновый режим)

**Получение статуса и результата из БД (опрашивает backend/клиент):**

- `GET /tasks/{task_id}` - статус и результат задачи (pending / processing / completed / failed)
- `GET /tasks?limit=50&offset=0` - список задач с пагинацией

### Структура проекта

```
whisper_video_summarization/
├── configs/              # Конфигурационные файлы Hydra
│   ├── train.yaml
│   ├── model/
│   ├── data/
│   ├── logging/
│   └── paths/
├── whisper_video_summarization/
│   ├── api/              # FastAPI приложение
│   ├── data/             # Датасеты и их обработка
│   ├── models/           # Модели (Whisper, T5)
│   ├── training/         # Скрипты обучения
│   ├── whisper/          # Модуль транскрипции
│   ├── streamlit/        # (устарел) ранее Streamlit интерфейс
│   └── utils/            # Утилиты
├── frontend/             # React + TypeScript (Vite) интерфейс
├── data/                 # Данные проекта
├── Dockerfile            # образ API (FastAPI, без PyTorch)
├── Dockerfile.worker     # образ Celery + Whisper/GPU
├── Dockerfile.mlflow     # образ MLflow UI
├── k8s/                  # Kubernetes: Postgres, RabbitMQ, API, worker, frontend, ingress
├── dvc.yaml              # DVC pipeline
├── pyproject.toml        # Poetry конфигурация
└── README.md
```

### Требования

- **Kubernetes** (или локальная разработка без кластера: Poetry + `npm run dev` для фронта)
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

Откройте http://localhost:5173. Запросы к `/api` будут проксироваться на http://127.0.0.1:8000 (запустите FastAPI отдельно).
