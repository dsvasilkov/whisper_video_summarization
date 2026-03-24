import logging
import os
import subprocess
from pathlib import Path

from whisper_video_summarization.utils.paths import get_paths

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_whisper_model_dir() -> Path:
    paths = get_paths()
    return PROJECT_ROOT / paths.whisper_model_dir


def run_dvc(cmd: list[str], check: bool = True) -> str:
    result = subprocess.run(
        ["dvc", *cmd],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout


def add_whisper_to_dvc():
    whisper_model_dir = get_whisper_model_dir()
    if not whisper_model_dir.exists():
        return

    files = [
        f
        for f in whisper_model_dir.iterdir()
        if not f.name.startswith(".") and not f.name.startswith("__")
    ]
    if not files:
        return

    whisper_relative = str(whisper_model_dir.relative_to(PROJECT_ROOT))
    dvc_file = PROJECT_ROOT / f"{whisper_relative}.dvc"
    if dvc_file.exists():
        return

    try:
        run_dvc(["add", whisper_relative], check=False)
    except Exception:
        pass


def dvc_pull():
    return run_dvc(["pull"], check=False)


def dvc_repro(stage: str):
    return run_dvc(["repro", stage])


def track_path_in_dvc(path: Path, push: bool = False) -> bool:
    """
    Регистрирует файл или каталог в DVC (путь должен быть внутри PROJECT_ROOT).
    Нужен инициализированный git-репозиторий в корне проекта.
    """
    try:
        path = path.resolve()
        rel = path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        logger.warning("DVC: путь %s вне PROJECT_ROOT, пропуск", path)
        return False
    if not path.exists():
        logger.warning("DVC: файл не найден: %s", path)
        return False
    git_dir = PROJECT_ROOT / ".git"
    if not git_dir.exists():
        logger.warning("DVC: нет каталога .git в %s — dvc add пропущен", PROJECT_ROOT)
        return False
    out = run_dvc(["add", str(rel)], check=False)
    if out:
        logger.debug("dvc add: %s", out.strip()[:500])
    if push or os.getenv("DVC_PUSH_UPLOADS", "").lower() in ("1", "true", "yes"):
        run_dvc(["push", str(rel)], check=False)
    return True
