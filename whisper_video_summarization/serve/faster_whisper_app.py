from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from ray import serve
from ray.serve import metrics

logger = logging.getLogger(__name__)

api = FastAPI(title="Faster-Whisper ASR (Ray Serve)")


def _env(name: str, default: str) -> str:
    return str(os.getenv(name, default)).strip()


def _env_positive_int(name: str, default: int) -> int:
    raw = _env(name, "")
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def _env_optional_positive_int(name: str) -> int | None:
    raw = _env(name, "")
    if not raw:
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except ValueError:
        return None


def _clip_chunk_seconds() -> int:
    """Макс. длина клипа (сек): VAD + collect_chunks, как max_clip_duration_s у vLLM."""
    v = _env_optional_positive_int("WHISPER_CHUNK_LENGTH")
    if v is not None:
        return v
    v2 = _env_optional_positive_int("WHISPER_VAD_MAX_SPEECH_DURATION_S")
    if v2 is not None:
        return v2
    return 30


DEFAULT_MODEL = "Systran/faster-whisper-large-v3"
_UPLOAD_CHUNK = 1024 * 1024

# Ray-native counters: попадают в стандартный /metrics на прокси Serve (см. job=ray-serve в Prometheus).
_whisper_serve_input_audio_seconds = metrics.Counter(
    "whisper_serve_input_audio_seconds",
    "Cumulative input audio duration (seconds) for completed /transcribe (server-side).",
)
_whisper_serve_inference_seconds = metrics.Counter(
    "whisper_serve_inference_seconds",
    "Cumulative wall-clock inference time in /transcribe after the body is fully read (seconds).",
)


def _probe_audio_duration_seconds(path: Path) -> float | None:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        return None


@serve.deployment(
    ray_actor_options={
        "num_cpus": float(_env("WHISPER_NUM_CPUS", "2") or "2"),
        "num_gpus": float(_env("WHISPER_NUM_GPUS", "0.34") or "0.34"),
    },
    max_ongoing_requests=int(_env("WHISPER_MAX_ONGOING_REQUESTS", "1") or "1"),
)
@serve.ingress(api)
class FasterWhisperAsr:
    def __init__(self) -> None:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
        from faster_whisper.vad import VadOptions

        model_id = _env("WHISPER_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
        device = _env("WHISPER_DEVICE", "cuda") or "cuda"
        compute_type = _env("WHISPER_COMPUTE_TYPE", "int8") or "int8"
        download_root = _env("WHISPER_DOWNLOAD_ROOT", "")
        self._chunk_s = _clip_chunk_seconds()
        self._batch_size = _env_positive_int("WHISPER_BATCH_SIZE", 4)
        self._vad_min_silence_ms = _env_positive_int("WHISPER_VAD_MIN_SILENCE_MS", 160)
        self._vad_options = VadOptions(
            max_speech_duration_s=float(self._chunk_s),
            min_silence_duration_ms=self._vad_min_silence_ms,
        )
        kwargs: dict[str, Any] = {}
        if download_root:
            kwargs["download_root"] = download_root

        logger.info(
            "Loading Faster-Whisper (VAD + batched decode): model=%s device=%s "
            "compute_type=%s chunk_s=%s batch_size=%s vad_min_silence_ms=%s",
            model_id,
            device,
            compute_type,
            self._chunk_s,
            self._batch_size,
            self._vad_min_silence_ms,
        )
        self._model = WhisperModel(
            model_id,
            device=device,
            compute_type=compute_type,
            **kwargs,
        )
        self._pipeline = BatchedInferencePipeline(self._model)
        self._model_id = model_id
        self._device = device
        self._compute_type = compute_type

    @api.get("/health")
    def health(self) -> dict[str, str]:
        return {
            "status": "ok",
            "model": self._model_id,
            "device": self._device,
            "compute_type": self._compute_type,
            "inference": "batched_vad",
            "chunk_s": str(self._chunk_s),
            "batch_size": str(self._batch_size),
            "vad_min_silence_ms": str(self._vad_min_silence_ms),
        }

    @api.post("/transcribe")
    async def transcribe(
        self,
        file: UploadFile = File(...),
        language: str = Form(""),
    ) -> dict[str, Any]:
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

            lang = language.strip() or None
            # Как vLLM: VAD → клипы ≤ chunk_s → признаки по окнам (не весь файл сразу) → батч на GPU.
            infer_started = time.perf_counter()
            segments_gen, _ = self._pipeline.transcribe(
                str(tmp_path),
                language=lang,
                word_timestamps=True,
                without_timestamps=False,
                temperature=[0.0],
                vad_filter=True,
                vad_parameters=self._vad_options,
                chunk_length=self._chunk_s,
                batch_size=self._batch_size,
            )
            raw_segments: list[dict[str, Any]] = []
            for seg in segments_gen:
                words_out: list[dict[str, Any]] = []
                words = getattr(seg, "words", None)
                if words:
                    for w in words:
                        words_out.append(
                            {
                                "start": float(w.start),
                                "end": float(w.end),
                                "word": w.word,
                            }
                        )
                raw_segments.append(
                    {
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": seg.text,
                        "words": words_out,
                    }
                )
            infer_elapsed = time.perf_counter() - infer_started
            input_sec = _probe_audio_duration_seconds(tmp_path)
            if input_sec is None and raw_segments:
                try:
                    input_sec = max(float(s.get("end", 0.0)) for s in raw_segments)
                except Exception:
                    input_sec = None
            if (
                input_sec is not None
                and input_sec > 0
                and infer_elapsed > 0
            ):
                _whisper_serve_input_audio_seconds.inc(input_sec)
                _whisper_serve_inference_seconds.inc(infer_elapsed)
            texts = [str(s.get("text", "")).strip() for s in raw_segments]
            full_text = " ".join(t for t in texts if t).strip()
            return {
                "text": full_text,
                "segments": raw_segments,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Transcription failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass


app = FasterWhisperAsr.bind()
