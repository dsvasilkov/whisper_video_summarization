from pathlib import Path

from hydra import compose, initialize


def get_paths():
    """Load paths from config."""
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="train")
    return cfg.paths


def get_path(key: str) -> Path:
    """Get a specific path from config."""
    paths = get_paths()
    return Path(getattr(paths, key))
