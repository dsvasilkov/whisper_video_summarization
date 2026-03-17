from collections.abc import Generator

from sqlalchemy.orm import Session

from whisper_video_summarization.db.session import get_session_factory


def get_db() -> Generator[Session, None, None]:
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
