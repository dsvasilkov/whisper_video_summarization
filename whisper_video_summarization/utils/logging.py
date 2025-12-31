import subprocess
from pytorch_lightning.loggers import MLFlowLogger


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_mlflow_logger(cfg):
    logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.uri,
    )

    logger.log_hyperparams(
        {
            "model": cfg.model.name,
            "lr": cfg.model.lr,
            "batch_size": cfg.train.batch_size,
            "epochs": cfg.train.epochs,
            "git_commit": get_git_commit(),
        }
    )

    return logger
