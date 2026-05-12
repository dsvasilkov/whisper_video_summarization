import logging
import os
import subprocess
from pathlib import Path

from whisper_video_summarization.utils.paths import get_paths

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VLLM_MODEL_PATHS: dict[str, str] = {
    "Systran/faster-whisper-large-v3": "vllm_asr_model_dir",
    "unsloth/Qwen3.5-9B-GGUF (Q8 GGUF)": "vllm_qwen_model_dir",
    "Qwen/Qwen3.5-9B (tokenizer+config)": "vllm_qwen_hf_tokenizer_dir",
}


def get_whisper_model_dir() -> Path:
    paths = get_paths()
    return PROJECT_ROOT / paths.whisper_model_dir


def get_vllm_model_dirs() -> list[Path]:
    paths = get_paths()
    dirs: list[Path] = []
    for key in VLLM_MODEL_PATHS.values():
        value = getattr(paths, key, None)
        if value:
            dirs.append(PROJECT_ROOT / value)
    return dirs


def run_dvc(cmd: list[str], check: bool = True) -> str:
    try:
        result = subprocess.run(
            ["dvc", *cmd],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=check,
        )
        return result.stdout
    except FileNotFoundError:
        logger.warning("DVC binary is not installed; skipping: dvc %s", " ".join(cmd))
        return ""


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


def add_vllm_models_to_dvc():
    for model_dir in get_vllm_model_dirs():
        if not model_dir.exists():
            continue

        files = [
            f
            for f in model_dir.iterdir()
            if not f.name.startswith(".") and not f.name.startswith("__")
        ]
        if not files:
            continue

        model_relative = str(model_dir.relative_to(PROJECT_ROOT))
        dvc_file = PROJECT_ROOT / f"{model_relative}.dvc"
        if dvc_file.exists():
            continue

        try:
            run_dvc(["add", model_relative], check=False)
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
