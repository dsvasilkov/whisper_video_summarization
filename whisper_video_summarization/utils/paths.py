from pathlib import Path

from hydra import compose, initialize


def get_paths():
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="infer")
    return cfg.paths


def get_path(key: str) -> Path:
    paths = get_paths()
    return Path(getattr(paths, key))
