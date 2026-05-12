import asyncio
import logging
import os
import time
from pathlib import Path
import httpx
from typing import Any

from whisper_video_summarization.utils.observability import (
    observe_pyannote_diarization,
    observe_whisper_inference,
)

logger = logging.getLogger(__name__)


# -----------------------------
# Config
# -----------------------------
def _asr_timeout_seconds() -> float:
    raw = os.getenv("WHISPER_ASR_TIMEOUT_SECONDS", "").strip()
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            pass
    return 3600.0


def _diarization_enabled() -> bool:
    return os.getenv("PYANNOTE_ENABLED", "").lower() in {"1", "true", "yes", "on"}


def _pyannote_serve_url() -> str | None:
    raw = os.getenv("PYANNOTE_SERVE_URL", "").strip().rstrip("/")
    return raw or None


def _whisper_serve_url() -> str | None:
    raw = os.getenv("WHISPER_SERVE_URL", "").strip().rstrip("/")
    return raw or None


def _diarization_tolerance_seconds() -> float:
    raw = os.getenv("PYANNOTE_ASSIGN_TOLERANCE_SEC", "0.35")
    try:
        return max(0.0, float(raw))
    except Exception:
        return 0.35


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


def _speaker_for_interval(
    t_start: float,
    t_end: float,
    speakers: list[dict[str, Any]],
) -> str | None:
    """Map an interval to a pyannote speaker by largest overlap; fallback to midpoint."""
    if not speakers:
        return None
    if t_end < t_start:
        t_start, t_end = t_end, t_start
    best: str | None = None
    best_ov = -1.0
    for sp in speakers:
        try:
            a = float(sp["start"])
            b = float(sp["end"])
        except Exception:
            continue
        ov = min(t_end, b) - max(t_start, a)
        if ov > best_ov:
            best_ov = ov
            best = str(sp["speaker"])
    if best is not None and best_ov > 0:
        return best
    mid = 0.5 * (t_start + t_end)
    tolerance = _diarization_tolerance_seconds()
    s = _strict_speaker_for_time(mid, speakers)
    if s is not None:
        return s
    return _nearest_speaker_for_time(mid, speakers, tolerance)


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
            ws = float(start)
            we = float(end)
            mid = (ws + we) / 2
            speaker = _strict_speaker_for_time(mid, speakers)
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
        item: dict[str, Any] = {
            "speaker": str(seg.get("speaker") or "Unknown"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "start_label": start,
            "end_label": end,
            "time_label": f"{start} - {end}",
            "text": seg_text,
        }
        # Must persist for Celery merge (diarization): word timestamps + per-word speaker polish.
        words = seg.get("words")
        if isinstance(words, list) and words:
            item["words"] = words
        payload_segments.append(item)

    full_text = _segments_to_text(segments)
    return {
        "format": "speaker_segments_v1",
        "segments": payload_segments,
        "text": full_text,
    }


# Celery/ASR путь отдаёт на serve WAV; multipart Content-Type для httpx.
_ASR_HTTP_UPLOAD_MEDIA_TYPE = "audio/wav"


def _parse_pyannote_serve_json(payload: Any) -> list[dict[str, Any]]:
    speakers = payload.get("speakers") if isinstance(payload, dict) else None
    if not isinstance(speakers, list):
        return []
    return [s for s in speakers if isinstance(s, dict)]


async def _run_pyannote_serve_async(audio_path: Path) -> tuple[list[dict[str, Any]], float]:
    base = _pyannote_serve_url()
    if not base:
        raise RuntimeError("PYANNOTE_SERVE_URL is not set")
    url = f"{base}/diarize"
    timeout = float(os.getenv("PYANNOTE_SERVE_TIMEOUT_SECONDS", "3600"))
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            with audio_path.open("rb") as fp:
                files = {"file": (audio_path.name, fp, _ASR_HTTP_UPLOAD_MEDIA_TYPE)}
                resp = await client.post(url, files=files)
        if resp.status_code != 200:
            raise RuntimeError(f"pyannote serve failed: {resp.status_code} {resp.text}")
        elapsed = time.perf_counter() - started
        return _parse_pyannote_serve_json(resp.json()), elapsed
    finally:
        observe_pyannote_diarization(duration_seconds=time.perf_counter() - started)


async def _run_whisper_serve_async(
    audio_path: Path,
    language: str,
    *,
    input_seconds: float | None = None,
) -> tuple[dict[str, Any], float]:
    base = _whisper_serve_url()
    if not base:
        raise RuntimeError("WHISPER_SERVE_URL is not set")
    url = f"{base}/transcribe"
    timeout = _asr_timeout_seconds()
    data = {"language": (language or "").strip()}
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        with audio_path.open("rb") as fp:
            files = {"file": (audio_path.name, fp, _ASR_HTTP_UPLOAD_MEDIA_TYPE)}
            resp = await client.post(url, files=files, data=data)
    elapsed = time.perf_counter() - started
    if resp.status_code != 200:
        raise RuntimeError(f"Whisper Serve failed: {resp.status_code} {resp.text}")
    body = resp.json()
    if isinstance(body, dict):
        return body, elapsed
    return {}, elapsed


async def _maybe_get_speakers(
    audio_path: Path,
    diarize: bool | None = None,
) -> tuple[list[dict[str, Any]], bool, float]:
    """Returns (speaker segments, skipped_external_service).

    ``skipped_external_service`` is True when diarization was requested but Ray/pyannote
    HTTP could not be used (misconfig or error); UI uses this via ``payload._meta``."""
    enabled = _diarization_enabled() if diarize is None else diarize
    if not enabled:
        return [], False, 0.0

    if not _pyannote_serve_url():
        logger.warning("Diarization skipped: PYANNOTE_SERVE_URL is not set")
        return [], True, 0.0

    path = Path(audio_path)
    try:
        speakers, py_elapsed = await _run_pyannote_serve_async(path)
        logger.info("Diarization completed: %d speaker segments", len(speakers))
        return speakers, False, py_elapsed
    except Exception as exc:
        logger.warning("Diarization skipped: %s", exc)
        return [], True, 0.0


def _inject_words_from_global_words(
    raw_segments: list[dict[str, Any]],
    global_words: list[dict[str, Any]] | None,
) -> None:
    """If the ASR API puts word timestamps in a top-level `words` array, map them onto segments."""
    if not global_words or not raw_segments:
        return
    for seg in raw_segments:
        if isinstance(seg.get("words"), list) and seg["words"]:
            continue
        try:
            s0 = float(seg.get("start", 0.0))
            s1 = float(seg.get("end", s0))
        except Exception:
            continue
        wds: list[dict[str, Any]] = []
        for w in global_words:
            if not isinstance(w, dict):
                continue
            try:
                ws = float(w.get("start", 0.0))
                we = float(w.get("end", ws))
            except Exception:
                continue
            if we <= s0 or ws >= s1:
                continue
            wds.append(dict(w))
        if wds:
            seg["words"] = wds


def _segments_to_text(segments: list[dict[str, Any]]) -> str:
    texts = [str(seg.get("text", "")).strip() for seg in segments]
    return " ".join(t for t in texts if t).strip()


def _normalize_asr_segments(
    raw_segments: list[dict[str, Any]],
    *,
    input_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Normalize ASR segments into a stable shape.

    For very long audio, some backends may intermittently emit timestamps relative to an
    internal chunk window (i.e. timestamps "jump backwards"). We unwrap the timeline
    by applying a running shift when we detect a large backward jump.
    """
    normalized: list[dict[str, Any]] = []
    timeline_shift = 0.0
    last_end = 0.0

    for seg in raw_segments:
        try:
            base_start = float(seg.get("start", 0.0))
            base_end = float(seg.get("end", base_start))
        except Exception:
            continue

        start_raw = max(base_start, 0.0)
        end_raw = max(base_end, 0.0)
        if end_raw < start_raw:
            end_raw = start_raw

        # Unwrap timeline on large backward jumps (likely timestamp reset).
        start_candidate = start_raw + timeline_shift
        if start_candidate + 5.0 < last_end:
            timeline_shift = last_end - start_raw

        start = start_raw + timeline_shift
        end = end_raw + timeline_shift
        if end < start:
            end = start

        words_out: list[dict[str, Any]] = []
        for word in seg.get("words", []):
            if not isinstance(word, dict):
                continue
            out_word = dict(word)
            try:
                ws = max(float(out_word.get("start")), 0.0)
                out_word["start"] = ws + timeline_shift
            except Exception:
                pass
            try:
                we = max(float(out_word.get("end")), 0.0)
                out_word["end"] = we + timeline_shift
            except Exception:
                pass
            words_out.append(out_word)

        normalized.append(
            {
                "start": start,
                "end": end,
                "text": seg.get("text", ""),
                "words": words_out,
            }
        )
        last_end = max(last_end, end)

    # If the produced timestamps drift beyond the real media duration (common with
    # sample-rate mismatches or backend quirks), rescale to fit the actual duration.
    if (
        input_seconds is not None
        and input_seconds > 0
        and last_end > 0
        and last_end > input_seconds * 1.001
    ):
        scale = input_seconds / last_end

        def _clamp(v: float) -> float:
            return max(0.0, min(v, input_seconds))

        for seg in normalized:
            try:
                seg["start"] = _clamp(float(seg.get("start", 0.0)) * scale)
            except Exception:
                seg["start"] = 0.0
            try:
                seg["end"] = _clamp(float(seg.get("end", seg["start"])) * scale)
            except Exception:
                seg["end"] = seg["start"]
            if seg["end"] < seg["start"]:
                seg["end"] = seg["start"]

            words = seg.get("words", [])
            if isinstance(words, list):
                for w in words:
                    if not isinstance(w, dict):
                        continue
                    try:
                        w["start"] = _clamp(max(float(w.get("start", 0.0)), 0.0) * scale)
                    except Exception:
                        pass
                    try:
                        w["end"] = _clamp(max(float(w.get("end", 0.0)), 0.0) * scale)
                    except Exception:
                        pass

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
    for word in seg.get("words", []):
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

    by_interval = _speaker_for_interval(seg_start_f, seg_end_f, speakers)
    if by_interval is not None:
        return by_interval
    mid = (seg_start_f + seg_end_f) / 2
    tolerance = _diarization_tolerance_seconds()
    strict = _strict_speaker_for_time(mid, speakers)
    if strict is not None:
        return strict
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


def _finalize_asr_payload(
    raw_segments: list[dict[str, Any]],
    *,
    fallback_plain: str,
    language: str,
    speakers: list[dict[str, Any]],
    started: float,
    input_seconds: float | None,
    run_diarize: bool,
    diarization_skipped: bool,
    whisper_wall_seconds: float,
    pyannote_wall_seconds: float,
) -> dict[str, Any]:
    segments = _normalize_asr_segments(raw_segments, input_seconds=input_seconds)
    text_from_segments = _segments_to_text(segments)
    text = text_from_segments or fallback_plain
    if not segments and fallback_plain:
        segments = [{"start": 0.0, "end": 0.0, "text": fallback_plain, "words": []}]

    if speakers:
        segments = _assign_speakers(segments, speakers)
    segments = _assign_segment_speakers(segments, speakers)

    text = text or _segments_to_text(segments)
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
    payload["language"] = language

    meta: dict[str, Any] = {
        "asr_done": True,
        "diarization_ready": True,
        "merge_done": True,
        "task_wall_whisper_seconds": max(float(whisper_wall_seconds), 0.0),
        "task_wall_pyannote_seconds": max(float(pyannote_wall_seconds), 0.0),
    }
    if run_diarize and diarization_skipped:
        meta["diarization_skipped"] = True
    prev_meta = payload.get("_meta")
    if isinstance(prev_meta, dict):
        merged = dict(prev_meta)
        merged.update(meta)
        payload["_meta"] = merged
    else:
        payload["_meta"] = meta

    elapsed = time.perf_counter() - started
    observe_whisper_inference(
        duration_seconds=elapsed,
        input_seconds=input_seconds,
        output_tokens=max(len(text.split()), 1),
    )
    return payload


def _probe_audio_duration_seconds(path: Path) -> float | None:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        return None


async def _transcribe_whole_file(
    audio_path: Path,
    language: str,
    diarize: bool | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_diarize = _effective_diarize(diarize)

    audio_path = Path(audio_path)
    input_seconds = await asyncio.to_thread(_probe_audio_duration_seconds, audio_path)
    timeout = _asr_timeout_seconds()

    if not _whisper_serve_url():
        raise RuntimeError("WHISPER_SERVE_URL is not set")

    diarization_skipped = False
    whisper_elapsed = 0.0
    pyannote_elapsed = 0.0
    if run_diarize:
        (result, whisper_elapsed), (speakers, diarization_skipped, pyannote_elapsed) = await asyncio.gather(
            asyncio.wait_for(
                _run_whisper_serve_async(audio_path, language, input_seconds=input_seconds),
                timeout=timeout,
            ),
            _maybe_get_speakers(audio_path, diarize=True),
        )
    else:
        result, whisper_elapsed = await asyncio.wait_for(
            _run_whisper_serve_async(audio_path, language, input_seconds=input_seconds),
            timeout=timeout,
        )
        speakers = []
        diarization_skipped = False

    if not isinstance(result, dict):
        result = {}
    fallback_plain = str(result.get("text") or "").strip()
    raw_segments = result.get("segments", [])
    if not isinstance(raw_segments, list):
        raw_segments = []
    raw_segments = [s for s in raw_segments if isinstance(s, dict)]

    return _finalize_asr_payload(
        raw_segments,
        fallback_plain=fallback_plain,
        language=language,
        speakers=speakers,
        started=started,
        input_seconds=input_seconds,
        run_diarize=run_diarize,
        diarization_skipped=diarization_skipped,
        whisper_wall_seconds=whisper_elapsed,
        pyannote_wall_seconds=pyannote_elapsed,
    )


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
    speakers, _, _ = await _maybe_get_speakers(audio_path, diarize=True)
    return speakers
