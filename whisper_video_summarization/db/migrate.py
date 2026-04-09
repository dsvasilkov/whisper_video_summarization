"""CLI entrypoint for DB migrations in Kubernetes Job / CI (`alembic upgrade head`)."""

from whisper_video_summarization.db import init_db


def main() -> None:
    init_db()


if __name__ == "__main__":
    main()
