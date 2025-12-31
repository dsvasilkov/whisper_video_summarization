import subprocess
from pathlib import Path

from whisper_video_summarization.utils.paths import get_paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_whisper_model_dir() -> Path:
    """Get whisper model directory from config."""
    paths = get_paths()
    return PROJECT_ROOT / paths.whisper_model_dir


def run_dvc(cmd: list[str], check: bool = True) -> str:
    """Run DVC command."""
    result = subprocess.run(
        ["dvc", *cmd],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout


def add_whisper_to_dvc():
    """Add whisper model directory to DVC if it exists, has files, and is not tracked."""
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
    """Pull models, data, and other DVC-tracked files."""
    return run_dvc(["pull"], check=False)


def dvc_repro(stage: str):
    """Run DVC repro for a specific stage."""
    return run_dvc(["repro", stage])
