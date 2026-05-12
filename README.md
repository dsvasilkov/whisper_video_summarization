# Whisper Video Summarization

Проект для автоматической суммаризации видео: **ASR** через **Ray Serve** (faster-whisper [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3), `int8`) или отдельный **vLLM** с аудио-моделью (см. `k8s/vllm-asr.yaml`); **LLM** — **Qwen** через **vLLM** OpenAI API. Celery-воркеры ставят задачи в очереди; ASR-воркер ходит по HTTP к сервису транскрипции; суммаризация и RAG — к vLLM и вспомогательным сервисам.

## Описание проекта

- **ASR** — [Ray Serve](https://docs.ray.io/en/latest/serve/index.html): faster-whisper + опционально **pyannote** (диаризация) в одном multi-app деплое (`Dockerfile.ray`, `k8s/ray-serve.yaml`). Альтернатива: деплой **`k8s/vllm-asr.yaml`** (тот же образ `whisper-vllm`, модель `Systran/faster-whisper-large-v3`).
- **Qwen (vLLM)** — суммаризация, граф тем / mind map, RAG; идентификатор модели задаётся в **`configs/model/qwen.yaml`**.
- **FastAPI** — постановка задач, статусы и результаты из БД, presign MinIO, webhook событий.
- **aio-celery + RabbitMQ** — очереди **`asr`**, **`llm`**, **`rag`**.
- **PostgreSQL + SQLAlchemy** — задачи и пользователи.
- **Redis** — `CELERY_RESULT_BACKEND` (ожидание RAG из API) и pub/sub для **SSE** статусов (`whisper_video_summarization/utils/task_events.py`; при необходимости отдельно **`TASK_EVENTS_REDIS_URL`**).
- **Qdrant** — векторный индекс для RAG (`k8s/qdrant.yaml`).
- **React + TypeScript (Vite)** — загрузка медиа, транскрипт, иерархия/лекция, mind map, Q&A.
- **Prometheus + Grafana** — `k8s/monitoring.yaml`, дашборд `k8s/grafana/whisper-overview.json`.
- **Hydra** — `configs/`; **DVC** — данные и модели (см. `whisper_video_summarization/utils/dvc.py`).

**Модели и пути (ориентир):**

- ASR (кэш): `Systran/faster-whisper-large-v3` → `data/models/vllm/Systran__faster-whisper-large-v3` (`configs/paths/paths.yaml`).
- LLM: **`configs/model/qwen.yaml`** — по умолчанию [`cyankiwi/Qwen3.5-2B-AWQ-4bit`](https://huggingface.co/cyankiwi/Qwen3.5-2B-AWQ-4bit). Поля `vllm_qwen_*` в `configs/paths/paths.yaml` относятся к локальному кэшу/legacy; сверяйте с актуальным `qwen.yaml`.

**Пайплайн суммаризации (кратко):** после ASR строится **unit graph** (Leiden и др.), при необходимости **иерархическое суммирование** (`llm/hierarchy_summarize.py`) и **mind map** (`llm/topic_graph_mindmap.py`) — см. `llm/infer.py`.

### Основные возможности

1. Транскрипция (Ray Serve или vLLM ASR по выбору деплоя).
2. Диаризация (pyannote в Ray, флаг **`PYANNOTE_ENABLED`** на API).
3. Суммаризация и визуализация структуры (темы, лекция, mind map).
4. RAG / вопросы по транскрипту (очередь **`rag`**).
5. Очередь и мониторинг (Prometheus/Grafana).

## Технические детали

### Развёртывание (Docker и Kubernetes)

В продакшене ориентир — **Kubernetes**. Docker Compose в репозитории не используется. Манифесты: **`k8s/`**, namespace **`whisper`**.

#### Требования

- Кластер с Ingress.
- Docker для сборки образов.
- GPU: драйвер NVIDIA, [GPU Operator / device plugin](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html); см. **`docs/gpu-docker-host.md`**.

#### Образы

| Тег | Dockerfile | Назначение |
|-----|--------------|------------|
| **whisper-api** | [`Dockerfile`](Dockerfile) | FastAPI; **uv** с **`UV_GROUP_FLAGS`** по умолчанию **`--group inference --group monitoring`** (без отдельного CUDA base-образа; PyTorch из lock для зависимостей inference на slim) |
| **whisper-worker** | [`Dockerfile.worker`](Dockerfile.worker) | Один образ для воркеров очередей **`asr` / `llm` / `rag`** (в K8s различаются командами **`-Q`** ) |
| **whisper-ray** | [`Dockerfile.ray`](Dockerfile.ray) | Ray Serve: faster-whisper, pyannote, embeddings (`k8s/ray-serve.yaml`) |
| **whisper-vllm** | [`Dockerfile.vllm`](Dockerfile.vllm) | vLLM OpenAI server + `vllm[audio]` — **`k8s/vllm-llm.yaml`** (LLM); тот же тег для **`k8s/vllm-asr.yaml`** при альтернативном ASR |
| **whisper-frontend** | [`frontend/Dockerfile`](frontend/Dockerfile) | Статика + nginx |

Подробнее: [`docs/docker-images.md`](docs/docker-images.md).

#### Сборка

```bash
eval $(minikube docker-env)   # пример для minikube
docker build -t whisper-api:latest .
docker build -f Dockerfile.worker -t whisper-worker:latest .
docker build -f Dockerfile.ray -t whisper-ray:latest .
docker build -f Dockerfile.vllm -t whisper-vllm:latest .
docker build -t whisper-frontend:latest ./frontend
```

Дополнительные имена тегов (`whisper-worker-asr` и т.д.) — тот же **`Dockerfile.worker`**, удобство для локальных скриптов.

#### Kubernetes: компоненты и переменные

- **Сервисы:** PostgreSQL, RabbitMQ, **Redis**, MinIO, API, **whisper-worker-asr**, **whisper-worker-llm**, **whisper-worker-rag**, **ray-serve** (образ `whisper-ray:latest`), **vllm-llm** (`whisper-vllm:latest`), опционально **vllm-asr**, **Qdrant**, Prometheus, Grafana, frontend, Ingress.
- **`DATABASE_URL`**, **`CELERY_BROKER_URL`**, **`CELERY_RESULT_BACKEND`** (Redis).
- **`S3_*`**, **`S3_PRESIGN_ENDPOINT_URL`** — MinIO и presign для браузера.
- ASR: **`WHISPER_SERVE_URL`** (Ray `/whisper` или URL vLLM ASR); **`PYANNOTE_SERVE_URL`**, **`PYANNOTE_ENABLED`** (API), **`PYANNOTE_HF_TOKEN`** (Ray, секрет в `ray-serve.yaml`).
- LLM/RAG: **`VLLM_LLM_BASE_URL`**, **`RAG_EMBEDDINGS_SERVE_URL`**, **`QDRANT_URL`**.

**Celery** — с явной очередью, как в манифестах:

```bash
aio_celery worker whisper_video_summarization.celery_app.app:celery_app -Q asr  -l INFO --concurrency=1
aio_celery worker whisper_video_summarization.celery_app.app:celery_app -Q llm  -l INFO --concurrency=1
aio_celery worker whisper_video_summarization.celery_app.app:celery_app -Q rag  -l INFO --concurrency=1
```

**Ingress:** UI на корне, бэкенд за префиксом **`/api`** (см. `frontend/nginx.conf`). Корневое приложение монтирует API: `app.mount("/api", api_app)` в `whisper_video_summarization/api/app.py`.

```bash
kubectl apply -f k8s/
```

Пример хоста: `whisper.local` в `k8s/ingress.yaml`.

### API и очереди

Все пути ниже с префиксом **`/api`** (полный URL вида `https://<host>/api/...`).

| Метод | Путь | Описание |
|--------|------|----------|
| POST | `/api/uploads/audio/presign` | Presign загрузки WAV в S3/MinIO |
| POST | `/api/minio/events` | Webhook `ObjectCreated:*` → постановка ASR |
| GET | `/api/tasks/{task_id}` | Статус и результат |
| GET | `/api/tasks` | Список (`limit`, `offset`, опционально `include_results`) |
| GET | `/api/tasks/{task_id}/events` | SSE обновлений (нужен Redis) |
| POST | `/api/tasks/{task_id}/qa` | Вопрос по задаче → очередь **`rag`** |
| GET | `/api/tasks/{task_id}/chunks/embeddings` | Эмбеддинги чанков → **`rag`** |
| POST | `/api/auth/register`, `/api/auth/login`, `/api/auth/forgot-password`, `/api/auth/reset-password` | Аутентификация |

**Очереди:** **`asr`** (транскрипт + опционально диаризация), **`llm`** (суммаризация после ASR), **`rag`** (индексация, QA, эмбеддинги для UI).

**Поток загрузки:** ffmpeg.wasm → WAV → `POST .../presign` (sha256 WAV) → PUT в MinIO → событие → **`/api/minio/events`** → очередь **`asr`**. Клиент: **`GET /api/tasks/{id}`** или SSE **`/api/tasks/{id}/events`**.

### Структура репозитория

```
configs/                          # Hydra
whisper_video_summarization/
  api/                            # FastAPI
  celery_app/                     # app, tasks, tasks_rag
  db/                             # модели SQLAlchemy, миграции
  llm/                            # infer, RAG, unit_graph, hierarchy_summarize, topic_graph_mindmap, …
  serve/                          # Ray Serve: faster_whisper, pyannote, embeddings
  whisper/                        # клиентская логика к ASR (HTTP и т.д.)
  utils/                          # S3, метрики, task_events, …
frontend/
Dockerfile
Dockerfile.worker
Dockerfile.ray
Dockerfile.vllm
k8s/
pyproject.toml
README.md
```

### Требования

- Kubernetes для продакшена **или** локально: Python **≥3.12** (`pyproject.toml`), uv, отдельно **`uvicorn`** для API и **`npm run dev`** для фронта.

### Данные и тома

- **MinIO** (`k8s/minio.yaml`): прямая загрузка браузера по presign; воркер скачивает объект во временный файл. **`S3_PRESIGN_ENDPOINT_URL`** должен быть доступен клиенту (часто отдельный Ingress, см. `k8s/ingress.yaml`).
- Уведомления бакета на **`/api/minio/events`**; при presign API может настроить notifications (ARN по умолчанию `arn:minio:sqs::api:webhook`, переопределение **`MINIO_WEBHOOK_QUEUE_ARN`**).
- **PVC `whisper-uploads` → `/app/data`** — HF-кэш, данные воркеров llm/rag/vLLM по манифестам.

Локально: **`data/`**, **`configs/`**; при необходимости MLflow (`mlruns/`, `mlflow.db`).

## Локальная разработка фронтенда

```bash
cd frontend
npm install
npm run dev
```

Порт **5173**, прокси **`/api`** → `http://127.0.0.1:8000` (`frontend/vite.config.ts`). Запуск API, например:

```bash
uv run uvicorn whisper_video_summarization.api.app:app --host 127.0.0.1 --port 8000
```

Первый запуск конвертации в браузере может быть медленнее из‑за загрузки **ffmpeg.wasm**.
