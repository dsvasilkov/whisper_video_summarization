import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from whisper_video_summarization.utils.observability import observe_whisper_inference

logger = logging.getLogger(__name__)

_diarization_executor: ThreadPoolExecutor | None = None
_pyannote_pipeline: Any | None = None
_pyannote_pipeline_lock = threading.Lock()
_pyannote_pipeline_key: tuple[str, str] | None = None


# -----------------------------
# Config
# -----------------------------
def _base_url() -> str:
    return os.getenv("VLLM_ASR_BASE_URL", "http://localhost:8001/v1")


def _diarization_enabled() -> bool:
    return os.getenv("PYANNOTE_ENABLED", "").lower() in {"1", "true", "yes", "on"}


def _diarization_token() -> str | None:
    return os.getenv("PYANNOTE_HF_TOKEN")


def _diarization_model() -> str:
    return os.getenv("PYANNOTE_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")


def _diarization_tolerance_seconds() -> float:
    raw = os.getenv("PYANNOTE_ASSIGN_TOLERANCE_SEC", "0.35")
    try:
        return max(0.0, float(raw))
    except Exception:
        return 0.35


def _timestamp_offset_seconds() -> float:
    raw = os.getenv("ASR_TIMESTAMP_OFFSET_SEC", "0")
    try:
        return float(raw)
    except Exception:
        return 0.0


def _diarization_executor_pool() -> ThreadPoolExecutor:
    global _diarization_executor
    if _diarization_executor is None:
        max_workers = max(1, int(os.getenv("PYANNOTE_MAX_WORKERS", "2")))
        _diarization_executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="pyannote",
        )
    return _diarization_executor


def _strict_speaker_for_time(
    time_point: float,
    speakers: list[dict[str, Any]],
) -> str | None:
    for sp in speakers:
        try:
            if float(sp["start"]) <= time_point <= float(sp["end"]):
                return str(sp["speaker"])
        except Exception:
            continue
    return None


def _nearest_speaker_for_time(
    time_point: float,
    speakers: list[dict[str, Any]],
    tolerance: float,
) -> str | None:
    nearest: str | None = None
    nearest_distance = float("inf")
    for sp in speakers:
        try:
            sp_start = float(sp["start"])
            sp_end = float(sp["end"])
        except Exception:
            continue
        if time_point < sp_start:
            distance = sp_start - time_point
        elif time_point > sp_end:
            distance = time_point - sp_end
        else:
            distance = 0.0
        if distance < nearest_distance:
            nearest_distance = distance
            nearest = str(sp["speaker"])
    if nearest is not None and nearest_distance <= tolerance:
        return nearest
    return None


def _assign_speakers(
    segments: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tolerance = _diarization_tolerance_seconds()
    for seg in segments:
        for w in seg.get("words", []):
            start = w.get("start")
            end = w.get("end")
            if start is None or end is None:
                w["speaker"] = None
                continue

            mid = (float(start) + float(end)) / 2
            # Prefer strict match in diarization boundaries.
            speaker = _strict_speaker_for_time(mid, speakers)
            # If pyannote has tiny gaps between turns, bind to the nearest turn
            # within a small tolerance instead of dropping to Unknown.
            if speaker is None:
                speaker = _nearest_speaker_for_time(mid, speakers, tolerance)
            w["speaker"] = speaker
    return segments


def _build_transcription_payload(segments: list[dict[str, Any]]) -> dict[str, Any]:
    payload_segments: list[dict[str, Any]] = []
    for seg in segments:
        seg_text = str(seg.get("text", "")).strip()
        if not seg_text:
            continue

        start = _format_timestamp(seg.get("start"))
        end = _format_timestamp(seg.get("end"))
        payload_segments.append(
            {
                "speaker": str(seg.get("speaker") or "Unknown"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "start_label": start,
                "end_label": end,
                "time_label": f"{start} - {end}",
                "text": seg_text,
            }
        )

    full_text = _segments_to_text(segments)
    return {
        "format": "speaker_segments_v1",
        "segments": payload_segments,
        "text": full_text,
    }


def _extract_speaker_ranges(pyannote_result: Any) -> list[dict[str, Any]]:
    annotation = pyannote_result
    # pyannote versions can return either Annotation directly or a wrapper
    # object (e.g. DiarizeOutput) that stores annotation-like attributes.
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


def _get_pyannote_pipeline_sync() -> Any:
    """Load pyannote once per worker process; from_pretrained on every task dominated runtime."""
    global _pyannote_pipeline, _pyannote_pipeline_key
    token = _diarization_token()
    if not token:
        raise RuntimeError("PYANNOTE_HF_TOKEN is not set")
    model_id = _diarization_model()
    key = (model_id, token)
    if _pyannote_pipeline is not None and _pyannote_pipeline_key == key:
        return _pyannote_pipeline
    with _pyannote_pipeline_lock:
        if _pyannote_pipeline is not None and _pyannote_pipeline_key == key:
            return _pyannote_pipeline
        try:
            from pyannote.audio import Pipeline
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pyannote.audio is not installed") from exc
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torch is not installed") from exc
        logger.info("Loading pyannote pipeline %s (one-time per process)", model_id)
        pipeline = Pipeline.from_pretrained(model_id, token=token)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        _pyannote_pipeline = pipeline
        _pyannote_pipeline_key = key
        return pipeline


def _run_pyannote_sync(audio_path: Path) -> list[dict[str, Any]]:
    pipeline = _get_pyannote_pipeline_sync()
    result = pipeline(str(audio_path))
    return _extract_speaker_ranges(result)


async def _maybe_get_speakers(
    audio_path: Path,
    diarize: bool | None = None,
) -> list[dict[str, Any]]:
    enabled = _diarization_enabled() if diarize is None else diarize
    if not enabled:
        return []

    try:
        loop = asyncio.get_running_loop()
        path = Path(audio_path)
        speakers = await loop.run_in_executor(
            _diarization_executor_pool(),
            _run_pyannote_sync,
            path,
        )
        logger.info("Diarization completed: %d speaker segments", len(speakers))
        return speakers
    except Exception as exc:
        logger.warning("Diarization skipped: %s", exc)
        return []


def _segments_to_text(segments: list[dict[str, Any]]) -> str:
    texts = [str(seg.get("text", "")).strip() for seg in segments]
    return " ".join(t for t in texts if t).strip()


def _normalize_asr_segments(raw_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize ASR timestamps to a monotonic global timeline.

    Some ASR backends can return chunk-local timestamps (e.g. timeline resets to
    ~0 for the next chunk). This function stitches such chunks into a single
    monotonic timeline and applies optional global offset for A/V alignment.
    """
    normalized: list[dict[str, Any]] = []
    carry = 0.0
    last_end = 0.0
    offset = _timestamp_offset_seconds()

    for seg in raw_segments:
        try:
            base_start = float(seg.get("start", 0.0))
            base_end = float(seg.get("end", base_start))
        except Exception:
            continue

        start = base_start + carry
        end = base_end + carry

        # Detect hard timeline reset from backend chunking.
        if normalized and start + 0.25 < last_end and base_start <= 2.0:
            carry = last_end
            start = base_start + carry
            end = base_end + carry

        # Keep timeline monotonic in noisy edge-cases.
        if start < last_end:
            start = last_end
        if end < start:
            end = start

        words_out: list[dict[str, Any]] = []
        for word in seg.get("words", []):
            if not isinstance(word, dict):
                continue
            out_word = dict(word)
            try:
                ws = float(out_word.get("start"))
                out_word["start"] = max(ws + carry + offset, 0.0)
            except Exception:
                pass
            try:
                we = float(out_word.get("end"))
                out_word["end"] = max(we + carry + offset, 0.0)
            except Exception:
                pass
            words_out.append(out_word)

        normalized.append(
            {
                "start": max(start + offset, 0.0),
                "end": max(end + offset, 0.0),
                "text": seg.get("text", ""),
                "words": words_out,
            }
        )
        last_end = end

    return normalized


def _format_timestamp(value: Any) -> str:
    try:
        total = max(float(value), 0.0)
    except Exception:
        return "00:00"

    total_seconds = int(total)
    milliseconds = int(round((total - total_seconds) * 1000))
    if milliseconds == 1000:
        total_seconds += 1
        milliseconds = 0

    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _pick_speaker_for_segment(
    seg: dict[str, Any],
    speakers: list[dict[str, Any]],
) -> str | None:
    counts: dict[str, int] = {}
    words = seg.get("words", [])
    for word in words:
        speaker = word.get("speaker")
        if speaker:
            key = str(speaker)
            counts[key] = counts.get(key, 0) + 1
    if counts:
        return max(counts.items(), key=lambda item: item[1])[0]

    seg_start = seg.get("start")
    seg_end = seg.get("end")
    if seg_start is None or seg_end is None or not speakers:
        return None

    try:
        seg_start_f = float(seg_start)
        seg_end_f = float(seg_end)
    except Exception:
        return None

    best_overlap = 0.0
    best_speaker: str | None = None
    for sp in speakers:
        try:
            sp_start = float(sp["start"])
            sp_end = float(sp["end"])
        except Exception:
            continue
        overlap = min(seg_end_f, sp_end) - max(seg_start_f, sp_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = str(sp["speaker"])
    if best_speaker:
        return best_speaker

    tolerance = _diarization_tolerance_seconds()
    mid = (seg_start_f + seg_end_f) / 2
    return _nearest_speaker_for_time(mid, speakers, tolerance)


def _smooth_unknown_speakers(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for i, seg in enumerate(segments):
        current = str(seg.get("speaker") or "Unknown")
        if current != "Unknown":
            continue
        prev_speaker = None
        next_speaker = None
        for j in range(i - 1, -1, -1):
            candidate = str(segments[j].get("speaker") or "Unknown")
            if candidate != "Unknown":
                prev_speaker = candidate
                break
        for j in range(i + 1, len(segments)):
            candidate = str(segments[j].get("speaker") or "Unknown")
            if candidate != "Unknown":
                next_speaker = candidate
                break
        if prev_speaker and next_speaker and prev_speaker == next_speaker:
            seg["speaker"] = prev_speaker
    return segments


def _assign_segment_speakers(
    segments: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for seg in segments:
        seg["speaker"] = _pick_speaker_for_segment(seg, speakers) or "Unknown"
    return _smooth_unknown_speakers(segments)


def _effective_diarize(diarize: bool | None) -> bool:
    if diarize is None:
        return _diarization_enabled()
    return diarize


async def _post_vllm_transcription(
    client: httpx.AsyncClient,
    audio_path: Path,
    language: str,
) -> httpx.Response:
    url = f"{_base_url()}/audio/transcriptions"
    with audio_path.open("rb") as file_obj:
        files = {"file": (audio_path.name, file_obj, "audio/wav")}
        data = {
            "model": os.getenv("VLLM_WHISPER_MODEL", "openai/whisper-large-v3"),
            "language": language,
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
            "temperature": "0",
        }
        return await client.post(url, data=data, files=files)


async def _transcribe_whole_file(
    audio_path: Path,
    language: str,
    diarize: bool | None = None,
) -> dict[str, Any]:
    timeout = float(os.getenv("VLLM_ASR_TIMEOUT_SECONDS", "3600"))
    started = time.perf_counter()
    run_diarize = _effective_diarize(diarize)

    async with httpx.AsyncClient(timeout=timeout) as client:
        if run_diarize:
            resp, speakers = await asyncio.gather(
                _post_vllm_transcription(client, audio_path, language),
                _maybe_get_speakers(audio_path, diarize=True),
            )
        else:
            resp = await _post_vllm_transcription(client, audio_path, language)
            speakers = []

    if resp.status_code != 200:
        raise RuntimeError(f"ASR failed: {resp.text}")

    result = resp.json()
    raw_segments = result.get("segments", [])
    if not isinstance(raw_segments, list):
        raw_segments = []
    segments = _normalize_asr_segments(raw_segments)
    if speakers:
        segments = _assign_speakers(segments, speakers)
    segments = _assign_segment_speakers(segments, speakers)

    text = _segments_to_text(segments)
    if not text:
        raise RuntimeError("ASR transcription failed: empty response text.")

    payload = _build_transcription_payload(segments)
    if not payload["segments"]:
        payload["segments"] = [
            {
                "speaker": "Unknown",
                "start": 0.0,
                "end": 0.0,
                "start_label": "00:00",
                "end_label": "00:00",
                "time_label": "00:00 - 00:00",
                "text": text,
            }
        ]
    payload["text"] = payload.get("text") or text

    elapsed = time.perf_counter() - started
    observe_whisper_inference(
        duration_seconds=elapsed,
        input_seconds=None,
        output_tokens=max(len(text.split()), 1),
    )
    return payload


# -----------------------------
# Public API
# -----------------------------
async def transcribe_audio(
    audio_path: Path,
    language: str = "ru",
    diarize: bool | None = None,
) -> dict[str, Any]:
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    return await _transcribe_whole_file(audio_path, language, diarize=diarize)


async def diarize_audio(audio_path: Path) -> list[dict[str, Any]]:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    return await _maybe_get_speakers(audio_path, diarize=True)
