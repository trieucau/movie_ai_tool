"""
Audio transcription module using faster-whisper.
Extracts audio from video and generates timestamped transcript.
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass, asdict

from app.utils import get_logger, ensure_dir, find_ffmpeg
from app.config import config

# Reuse one local model per process (avoid reload every run)
_local_whisper_model = None
_local_whisper_key: Optional[tuple] = None

logger = get_logger(__name__)

# --- Register NVIDIA CUDA DLL directories at import time (Windows only) ---
# This must happen BEFORE faster_whisper / ctranslate2 is imported so that
# Windows can locate cublas64_12.dll, cudnn_ops_infer64_8.dll, etc.
if sys.platform == "win32":
    _cuda_dirs_registered = []
    # Collect candidate directories from all known site-packages paths
    _search_roots = list(sys.path)
    try:
        import site as _site
        _search_roots += _site.getsitepackages()
        _search_roots += [_site.getusersitepackages()]
    except Exception:
        pass

    _nvidia_subdirs = [
        Path("nvidia") / "cublas" / "bin",
        Path("nvidia") / "cuda_runtime" / "bin",
        Path("nvidia") / "cuda_cupti" / "bin",
        Path("nvidia") / "cudnn" / "bin",
        Path("nvidia") / "cufft" / "bin",
        Path("nvidia") / "curand" / "bin",
        Path("nvidia") / "cusolver" / "bin",
        Path("nvidia") / "cusparse" / "bin",
    ]

    for _root in set(_search_roots):
        for _sub in _nvidia_subdirs:
            _candidate = Path(_root) / _sub
            if _candidate.exists() and str(_candidate) not in _cuda_dirs_registered:
                try:
                    os.add_dll_directory(str(_candidate))
                    _cuda_dirs_registered.append(str(_candidate))
                except Exception:
                    pass

    if _cuda_dirs_registered:
        logger.debug(f"Registered {len(_cuda_dirs_registered)} CUDA DLL directories for GPU support")
    else:
        logger.debug("No NVIDIA CUDA DLL directories found in site-packages")


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: Optional[List[dict]] = None

    def to_dict(self) -> dict:
        return asdict(self)


def extract_audio(
    video_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to the input video file.
        output_path: Path for the output audio file (WAV).

    Returns:
        Path to the extracted audio file.
    """
    output_path = output_path or config.paths.temp_dir / "audio.wav"
    ensure_dir(output_path.parent)

    logger.info(f"Extracting audio from: {video_path}")

    cmd = [
        find_ffmpeg(), "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr}")

    logger.info(f"Audio extracted: {output_path}")
    return output_path


def _transcribe_with_groq_api(
    audio_path: Path,
    language: Optional[str],
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[TranscriptSegment]:
    """Cloud Whisper via Groq — typically much faster than local CPU."""
    from groq import Groq

    if progress_callback:
        progress_callback(0.38, "Transcribing via Groq API...")

    client = Groq(api_key=config.api.groq_api_key)
    model = config.api.groq_whisper_model

    with open(audio_path, "rb") as audio_file:
        kwargs = {
            "file": audio_file,
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language:
            kwargs["language"] = language
        result = client.audio.transcriptions.create(**kwargs)

    segs: List[TranscriptSegment] = []
    raw_segments = getattr(result, "segments", None) or []
    for seg in raw_segments:
        if isinstance(seg, dict):
            text = (seg.get("text") or "").strip()
            start = seg.get("start", 0)
            end = seg.get("end", 0)
        else:
            text = (getattr(seg, "text", None) or "").strip()
            start = getattr(seg, "start", 0)
            end = getattr(seg, "end", 0)
        if not text:
            continue
        segs.append(TranscriptSegment(
            start=round(float(start), 3),
            end=round(float(end), 3),
            text=text,
            words=None,
        ))

    if not segs and getattr(result, "text", None):
        segs.append(TranscriptSegment(start=0.0, end=0.0, text=result.text.strip()))

    logger.info(f"Groq transcribed {len(segs)} segments")
    return segs


def _get_local_whisper_model(model_size: str, device: str, compute_type: str):
    global _local_whisper_model, _local_whisper_key
    key = (model_size, device, compute_type)
    if _local_whisper_model is not None and _local_whisper_key == key:
        return _local_whisper_model
    from faster_whisper import WhisperModel
    _local_whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
    _local_whisper_key = key
    return _local_whisper_model


def transcribe_audio(
    audio_path: Path,
    language: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[TranscriptSegment]:
    """
    Transcribe audio using faster-whisper with word-level timestamps.

    Args:
        audio_path: Path to the audio file (WAV/MP3).
        language: Language code ('en', 'vi', etc.) or None for auto-detect.
        progress_callback: Optional callback(percent, status_msg).

    Returns:
        List of TranscriptSegment with timestamps.
    """
    cfg = config.whisper
    engine = os.getenv("TRANSCRIBE_ENGINE", "auto").lower()

    # Skip broken CUDA attempt when user forces CPU
    prefer_cpu = os.getenv("WHISPER_DEVICE", "").lower() == "cpu"

    if config.api.groq_api_key and engine in ("auto", "groq"):
        try:
            segs = _transcribe_with_groq_api(audio_path, language, progress_callback)
            if progress_callback:
                progress_callback(0.55, f"Transcription complete ({len(segs)} segments)")
            return segs
        except Exception as e:
            logger.warning(f"Groq transcription failed ({e}), using local Whisper...")

    try:
        from faster_whisper import WhisperModel  # noqa: F401
    except ImportError:
        raise ImportError("faster-whisper is not installed. Run: pip install faster-whisper")

    model_size = cfg.model_size
    if language and language.lower() == "vi":
        model_size = os.getenv("WHISPER_MODEL_VI", "small")

    device = "cpu" if prefer_cpu else cfg.device
    compute_type = "int8" if device == "cpu" else cfg.compute_type
    if device == "cpu" and cfg.model_size in ("medium", "large", "large-v3"):
        model_size = os.getenv("WHISPER_CPU_MODEL", "base")
    use_words = os.getenv("WHISPER_WORD_TIMESTAMPS", "false").lower() == "true"

    logger.info(f"Loading Whisper model: {model_size} on {device}")

    if progress_callback:
        progress_callback(0.35, f"Loading Whisper ({model_size})...")

    def _run_transcription(mdl, audio_path, language):
        segments_raw, info = mdl.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=use_words,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,
        )
        segs: List[TranscriptSegment] = []
        for seg in segments_raw:
            words = []
            if seg.words:
                for w in seg.words:
                    words.append({"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word})
            segs.append(TranscriptSegment(
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=seg.text.strip(),
                words=words,
            ))
        return segs, info

    try:
        model = _get_local_whisper_model(model_size, device, compute_type)
        logger.info(f"Transcribing: {audio_path}")
        if progress_callback:
            progress_callback(0.40, "Transcribing audio...")
        segments, info = _run_transcription(model, audio_path, language)
    except Exception as gpu_err:
        if device != "cpu":
            logger.warning(f"GPU transcription failed ({gpu_err}), retrying on CPU...")
            fallback_size = os.getenv("WHISPER_CPU_FALLBACK_MODEL", "base")
            device = "cpu"
            compute_type = "int8"
            model = _get_local_whisper_model(fallback_size, device, compute_type)
            logger.info(f"Transcribing on CPU ({fallback_size}): {audio_path}")
            if progress_callback:
                progress_callback(0.40, f"Transcribing ({fallback_size} CPU)...")
            segments, info = _run_transcription(model, audio_path, language)
        else:
            raise

    logger.info(f"Transcribed {len(segments)} segments. Detected language: {info.language}")

    if progress_callback:
        progress_callback(0.55, f"Transcription complete ({len(segments)} segments)")

    return segments


def save_transcript(
    segments: List[TranscriptSegment],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save transcript segments to JSON file.

    Args:
        segments: List of TranscriptSegment.
        output_path: Path to save the JSON file.

    Returns:
        Path to saved transcript file.
    """
    output_path = output_path or config.paths.temp_dir / "transcript.json"
    ensure_dir(output_path.parent)

    data = [seg.to_dict() for seg in segments]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Transcript saved: {output_path} ({len(segments)} segments)")
    return output_path


def load_transcript(transcript_path: Path) -> List[TranscriptSegment]:
    """
    Load transcript from JSON file.

    Args:
        transcript_path: Path to transcript JSON.

    Returns:
        List of TranscriptSegment.
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = []
    for item in data:
        segments.append(
            TranscriptSegment(
                start=item["start"],
                end=item["end"],
                text=item["text"],
                words=item.get("words"),
            )
        )

    return segments


def transcript_to_text(segments: List[TranscriptSegment]) -> str:
    """Combine all transcript segments into a single text string."""
    return " ".join(seg.text for seg in segments)
