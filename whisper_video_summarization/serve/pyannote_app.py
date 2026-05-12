from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from ray import serve

logger = logging.getLogger(__name__)

api = FastAPI(title="pyannote diarization (Ray Serve)")

_UPLOAD_CHUNK = 1024 * 1024


def _env(name: str, default: str) -> str:
    return str(os.getenv(name, default)).strip()


def _diarization_model() -> str:
    return _env("PYANNOTE_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1") or "pyannote/speaker-diarization-3.1"


def _hf_token() -> str:
    token = _env("PYANNOTE_HF_TOKEN", "")
    if not token:
        token = _env("HF_TOKEN", "")
    return token


def _instantiate_params() -> dict[str, Any]:
    params: dict[str, Any] = {}
    raw_ct = _env("PYANNOTE_CLUSTERING_THRESHOLD", "")
    if raw_ct:
        try:
            params["clustering"] = {"threshold": float(raw_ct)}
        except Exception:
            pass
    raw_mdo = _env("PYANNOTE_SEGMENTATION_MIN_DURATION_OFF", "")
    if raw_mdo:
        try:
            params["segmentation"] = {"min_duration_off": float(raw_mdo)}
        except Exception:
            pass
    return params


def _extract_speaker_ranges(pyannote_result: Any) -> list[dict[str, Any]]:
    annotation = pyannote_result
    if not hasattr(annotation, "itertracks"):
        for attr in ("speaker_diarization", "annotation", "diarization"):
            candidate = getattr(pyannote_result, attr, None)
            if candidate is not None and hasattr(candidate, "itertracks"):
                annotation = candidate
                break

    if not hasattr(annotation, "itertracks"):
        raise RuntimeError(
            f"Unsupported pyannote diarization output type: {type(pyannote_result).__name__}"
        )

    speakers: list[dict[str, Any]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        speakers.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )
    return speakers


@serve.deployment(
    ray_actor_options={
        # Ray Serve will still schedule on CPU-only nodes if GPU is absent;
        # k8s manifests should request GPU when desired.
        "num_cpus": float(_env("PYANNOTE_NUM_CPUS", "2") or "2"),
        "num_gpus": float(_env("PYANNOTE_NUM_GPUS", "1") or "1"),
    },
    max_ongoing_requests=int(_env("PYANNOTE_MAX_ONGOING_REQUESTS", "2") or "2"),
)
@serve.ingress(api)
class PyannoteDiarizer:
    def __init__(self) -> None:
        token = _hf_token()
        if not token:
            raise RuntimeError("PYANNOTE_HF_TOKEN (or HF_TOKEN) is not set")

        model_id = _diarization_model()
        logger.info("Loading pyannote pipeline %s", model_id)

        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(model_id, token=token)
        inst = _instantiate_params()
        if inst:
            try:
                pipeline.instantiate(inst)
                logger.info("pyannote pipeline.instantiate applied: %s", inst)
            except Exception as exc:
                logger.warning("pyannote pipeline.instantiate(%s) ignored: %s", inst, exc)

        import torch

        device_raw = _env("PYANNOTE_DEVICE", "auto").lower()
        if device_raw == "cpu":
            device = torch.device("cpu")
        elif device_raw.startswith("cuda"):
            device = torch.device(device_raw)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pipeline.to(device)
        self._pipeline = pipeline
        self._model_id = model_id
        self._device = str(device)

        logger.info("pyannote ready: model=%s device=%s", self._model_id, self._device)

    @api.get("/health")
    def health(self) -> dict[str, str]:
        return {"status": "ok", "model": self._model_id, "device": self._device}

    @api.post("/diarize")
    async def diarize(self, file: UploadFile = File(...)) -> dict[str, Any]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")

        suffix = Path(file.filename).suffix or ".wav"
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)
                nonempty = False
                try:
                    while chunk := await file.read(_UPLOAD_CHUNK):
                        nonempty = True
                        tmp.write(chunk)
                except Exception as exc:
                    raise HTTPException(
                        status_code=400, detail=f"Failed to read upload: {exc}"
                    ) from exc
            if not nonempty:
                raise HTTPException(status_code=400, detail="Empty upload")
            result = self._pipeline(str(tmp_path))
            speakers = _extract_speaker_ranges(result)
            return {"speakers": speakers}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Diarization failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass


app = PyannoteDiarizer.bind()

