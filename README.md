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

### Setup

Проект использует Docker Compose для управления окружением. Для настройки выполните следующие шаги:

#### Требования

- Docker
- Docker Compose

#### Шаги установки

1. **Клонирование репозитория**

```bash
git clone <repository-url>
cd whisper_video_summarization
```

2. **Сборка Docker образов**

```bash
docker-compose build
```

Это создаст образы для сервисов:

- `fastapi` - API (producer): приём запросов, постановка задач в очередь, выдача статусов из БД
- `celery_worker` - consumer: обработка задач инференса из RabbitMQ
- `postgres` - БД для статусов задач (SQLAlchemy)
- `rabbitmq` - брокер очереди для Celery
- `mlflow` - сервис для отслеживания экспериментов
- `frontend` - веб-интерфейс (React + TypeScript)

3. **Запуск всех сервисов**

```bash
docker-compose up
```

После запуска сервисы будут доступны по следующим адресам:

- **FastAPI**: http://localhost:8000
- **FastAPI Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:8080
- **Frontend (React)**: http://localhost:5173
- **RabbitMQ Management**: http://localhost:15672 (guest/guest)
- **PostgreSQL**: localhost:5432 (postgres/postgres, БД `whisper_inference`)

#### Production: Nginx reverse proxy (три образа)

Окружение для сервера: один вход через Nginx (порт 80), приложение в режиме Production (DEBUG=False), статика отдаётся отдельным образом.

- **Образ 1 — Nginx**: reverse proxy, единственная точка входа на порт 80; проксирует `/api/` на приложение и `/` на статику.
- **Образ 2 — Приложение**: FastAPI за uvicorn (ASGI) с `--workers 2`, без проброса портов наружу.
- **Образ 3 — Статика**: собранный React (Vite), отдаётся по proxy со второго контейнера.

Конфигурация Nginx: `nginx/nginx.conf` (и `nginx/Dockerfile`).

Запуск production:

```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build
```

После запуска доступ только по **http://localhost** (порт 80): главная страница — статика, API — по префиксу `/api/` (например `/api/tasks`, `/api/infer/video/upload`).

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

**Через веб-интерфейс (React):** откройте http://localhost:5173 → «Обучение», загрузите датасет и нажмите «Запустить обучение».

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

Эта модель используется для инференса. При использовании Docker данные сохраняются в volume, поэтому модель будет доступна после перезапуска контейнеров.

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
├── dvc.yaml              # DVC pipeline
├── pyproject.toml        # Poetry конфигурация
└── README.md
```

### Требования

- Docker
- Docker Compose

### Работа с данными в Docker

При использовании Docker Compose данные и модели монтируются как volumes, поэтому изменения сохраняются между перезапусками:

- `./data` - данные проекта (включая `test_train.jsonl`)
- `./whisper_video_summarization/models` - модели
- `./configs` - конфигурационные файлы
- `./mlflow.db` - база данных MLflow
- `./mlruns` - артефакты MLflow
- `./tmp` - временные файлы (видео для обработки)

## Локальная разработка фронтенда

Без Docker, с проксированием запросов к API:

```bash
cd frontend
npm install
npm run dev
```

Откройте http://localhost:5173. Запросы к `/api` будут проксироваться на http://127.0.0.1:8000 (запустите FastAPI отдельно).
