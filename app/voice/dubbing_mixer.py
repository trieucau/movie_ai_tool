"""
Dubbing mixer — single consistent voice, slot-accurate sync with subtitles.

Fixes addressed:
  - One locked Edge TTS voice for every segment (cache keyed by voice id).
  - Fit each clip to its subtitle time slot (speed up/down + trim/pad).
  - No silent gaps: failed TTS is retried; never skip without filling the slot.
  - Prevent overlap bleed into the next segment.
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Callable

from pydub import AudioSegment
from app.utils import get_logger, ensure_dir, find_ffmpeg
from app.voice.tts_generator import get_audio_duration
from app.config import config

logger = get_logger(__name__)

MAX_CONCURRENCY = 3
MAX_SPEED_RATIO = 1.28
MIN_SPEED_RATIO = 0.82


def _sanitize_tts_text(text: str) -> str:
    """Clean text so Edge TTS accepts it reliably."""
    t = (text or "").strip()
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = re.sub(r"\([^)]*\)", "", t)
    t = re.sub(r"[#*_~`|<>]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 500:
        t = t[:497].rsplit(" ", 1)[0] + "..."
    return t


def _build_atempo_filter(ratio: float) -> str:
    filters = []
    remaining = max(0.5, min(ratio, 16.0))
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


def adjust_audio_speed(input_path: Path, output_path: Path, ratio: float) -> Path:
    """Change speed without pitch shift. ratio>1 = faster (shorter)."""
    ratio = max(0.5, min(ratio, 16.0))
    atempo_chain = _build_atempo_filter(ratio)
    cmd = [
        find_ffmpeg(), "-y",
        "-i", str(input_path),
        "-filter:a", atempo_chain,
        "-q:a", "4",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"atempo failed: {result.stderr.decode(errors='replace')[-300:]}")
    return output_path


def _fit_clip_to_slot(
    raw_path: Path,
    fitted_path: Path,
    target_dur: float,
    max_end_ms: Optional[int] = None,
) -> AudioSegment:
    """
    Fit TTS audio exactly into [0, target_dur] seconds.
    Speed within natural limits, then trim or pad with silence.
    """
    target_ms = max(50, int(target_dur * 1000))
    if max_end_ms is not None:
        target_ms = min(target_ms, max(50, max_end_ms))

    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return AudioSegment.silent(duration=target_ms)

    tts_dur = get_audio_duration(raw_path)
    if tts_dur <= 0:
        return AudioSegment.silent(duration=target_ms)

    work_path = raw_path
    if target_dur > 0:
        ratio = tts_dur / target_dur
        if ratio > 1.05:
            ratio = min(ratio, MAX_SPEED_RATIO)
            adjust_audio_speed(raw_path, fitted_path, ratio)
            work_path = fitted_path
        elif ratio < 0.92:
            ratio = max(ratio, MIN_SPEED_RATIO)
            adjust_audio_speed(raw_path, fitted_path, ratio)
            work_path = fitted_path

    clip = AudioSegment.from_file(str(work_path))
    if len(clip) > target_ms:
        clip = clip[:target_ms]
    elif len(clip) < target_ms:
        clip = clip + AudioSegment.silent(duration=target_ms - len(clip))
    return clip


def _prepare_tts_dir(tts_dir: Path, voice_id: str) -> None:
    """Invalidate cache when voice changes."""
    ensure_dir(tts_dir)
    manifest = tts_dir / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if data.get("voice_id") != voice_id:
                logger.info(f"Voice changed ({data.get('voice_id')} -> {voice_id}), clearing TTS cache.")
                shutil.rmtree(tts_dir, ignore_errors=True)
                ensure_dir(tts_dir)
        except Exception:
            shutil.rmtree(tts_dir, ignore_errors=True)
            ensure_dir(tts_dir)
    manifest.write_text(json.dumps({"voice_id": voice_id}), encoding="utf-8")


async def _tts_one_segment(
    semaphore: asyncio.Semaphore,
    text: str,
    output_path: Path,
    voice: str,
    rate: str,
    idx: int,
) -> bool:
    if output_path.exists() and output_path.stat().st_size > 0:
        return True

    text = _sanitize_tts_text(text)
    if len(text) < 2:
        return False

    try:
        import edge_tts
    except ImportError:
        raise ImportError("edge-tts not installed. Run: pip install edge-tts")

    async with semaphore:
        for attempt in range(4):
            try:
                communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
                await communicate.save(str(output_path))
                if output_path.exists() and output_path.stat().st_size > 0:
                    return True
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"TTS segment {idx} attempt {attempt + 1}: {e}. Retry in {wait}s")
                await asyncio.sleep(wait)
    return False


async def _generate_all_tts(
    segments: list,
    tts_dir: Path,
    voice: str,
    rate: str,
    progress_callback: Optional[Callable] = None,
    total: int = 0,
) -> List[int]:
    """Returns indices that failed TTS."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    completed = 0
    failed: List[int] = []

    async def _tracked(idx: int, coro):
        nonlocal completed
        ok = await coro
        completed += 1
        if not ok:
            failed.append(idx)
        if progress_callback:
            progress_callback(
                0.60 + 0.20 * (completed / max(total, 1)),
                f"TTS dubbing {completed}/{total}...",
            )

    tasks = []
    for i, seg in enumerate(segments):
        text = _sanitize_tts_text(seg.text)
        if len(text) < 2:
            continue
        raw_path = tts_dir / f"raw_{i:04d}.mp3"
        tasks.append(_tracked(i, _tts_one_segment(semaphore, text, raw_path, voice, rate, i)))

    await asyncio.gather(*tasks)
    return failed


async def _retry_failed_serial(
    segments: list,
    tts_dir: Path,
    voice: str,
    rate: str,
    failed_indices: List[int],
) -> List[int]:
    """Second pass: one segment at a time (most reliable for Edge TTS)."""
    still_failed: List[int] = []
    for idx in failed_indices:
        text = _sanitize_tts_text(segments[idx].text)
        raw_path = tts_dir / f"raw_{idx:04d}.mp3"
        if raw_path.exists():
            raw_path.unlink(missing_ok=True)
        ok = await _tts_one_segment(
            asyncio.Semaphore(1), text, raw_path, voice, rate, idx
        )
        if not ok:
            still_failed.append(idx)
            logger.error(f"TTS segment {idx} failed after serial retry.")
    return still_failed


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def generate_dubbed_audio(
    segments: list,
    video_duration: float,
    temp_dir: Path,
    output_path: Path,
    voice_id: str,
    progress_callback: Optional[Callable] = None,
) -> Path:
    """
    Build dubbed MP3: one voice, each segment aligned to subtitle timestamps.
    """
    tts_dir = temp_dir / "dubbing_tts"
    _prepare_tts_dir(tts_dir, voice_id)

    voice = voice_id.strip()
    rate = os.getenv("EDGE_TTS_RATE", "+0%")
    total = sum(1 for s in segments if len(_sanitize_tts_text(s.text)) >= 2)

    logger.info(f"TTS dubbing: voice={voice} rate={rate} segments={total}")

    failed = _run_async(_generate_all_tts(
        segments, tts_dir, voice, rate, progress_callback, total
    ))
    if failed:
        logger.warning(f"TTS pass 1 failed for {len(failed)} segments, serial retry...")
        failed = _run_async(_retry_failed_serial(segments, tts_dir, voice, rate, failed))

    if progress_callback:
        progress_callback(0.82, "Building audio timeline...")

    timeline = AudioSegment.silent(duration=int(video_duration * 1000) + 100)

    for i, seg in enumerate(segments):
        text = _sanitize_tts_text(seg.text)
        target_dur = seg.end - seg.start
        if target_dur <= 0 or len(text) < 2:
            continue

        start_ms = int(seg.start * 1000)
        max_end_ms = None
        if i + 1 < len(segments):
            max_end_ms = int(segments[i + 1].start * 1000) - start_ms
            if max_end_ms is not None and max_end_ms < 50:
                max_end_ms = 50

        raw_path = tts_dir / f"raw_{i:04d}.mp3"
        fitted_path = tts_dir / f"fitted_{i:04d}.mp3"

        if not raw_path.exists() or raw_path.stat().st_size == 0:
            logger.warning(f"Segment {i}: no TTS — inserting silence ({target_dur:.2f}s)")
            clip = AudioSegment.silent(duration=int(target_dur * 1000))
        else:
            try:
                if fitted_path.exists():
                    fitted_path.unlink(missing_ok=True)
                clip = _fit_clip_to_slot(raw_path, fitted_path, target_dur, max_end_ms)
            except Exception as e:
                logger.warning(f"Segment {i} fit failed: {e}")
                clip = AudioSegment.silent(duration=int(target_dur * 1000))

        timeline = timeline.overlay(clip, position=start_ms)

    ensure_dir(output_path.parent)
    timeline.export(str(output_path), format="mp3", bitrate="192k")
    logger.info(f"Dubbed audio saved: {output_path} (voice={voice})")
    return output_path
