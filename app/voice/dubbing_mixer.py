"""
Dubbing mixer module.
Generates TTS for all segments in PARALLEL (async), adjusts speed to fit
original time slots using chained atempo filters, then places them on an
exact timeline using pydub.

Performance notes
-----------------
- Edge TTS is a network call (~2-5s each). Serial processing of 300 segments
  would take 10-25 minutes. We use asyncio.gather with a semaphore to run up
  to MAX_CONCURRENCY requests at the same time, reducing total time by ~8-10x.
- Already-generated segment files are re-used (cache), so interrupted runs
  can resume without re-requesting TTS for segments that completed.
- atempo only accepts values in [0.5, 2.0]. For ratios outside that range we
  chain multiple atempo filters automatically.
"""

import asyncio
import subprocess
from pathlib import Path
from typing import List, Optional, Callable

from pydub import AudioSegment
from app.utils import get_logger, ensure_dir, find_ffmpeg
from app.voice.tts_generator import get_audio_duration
from app.config import config

logger = get_logger(__name__)

# Max concurrent Edge TTS connections. Edge TTS rate-limits per IP;
# 8 is a safe value that avoids 429s while being ~8x faster than serial.
MAX_CONCURRENCY = 8


def _build_atempo_filter(ratio: float) -> str:
    """
    Build a safe ffmpeg atempo filter chain for any ratio.
    atempo only accepts [0.5, 2.0], so we chain multiple stages.
    e.g. ratio=3.5 → 'atempo=2.0,atempo=1.75'
    """
    filters = []
    remaining = ratio
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    if remaining < 0.5:
        remaining = 0.5
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


def adjust_audio_speed(input_path: Path, output_path: Path, ratio: float) -> Path:
    """Use ffmpeg atempo chain to change audio speed without pitch shift."""
    ratio = max(0.5, min(ratio, 16.0))  # hard clamp — beyond 16x is unintelligible
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


async def _tts_one_segment(
    semaphore: asyncio.Semaphore,
    text: str,
    output_path: Path,
    voice: str,
    rate: str,
    idx: int,
) -> bool:
    """
    Async TTS for a single segment, guarded by semaphore.
    Skips if output already exists (cache / resume).
    Returns True on success, False on failure.
    """
    if output_path.exists() and output_path.stat().st_size > 0:
        return True  # cached

    try:
        import edge_tts
    except ImportError:
        raise ImportError("edge-tts not installed. Run: pip install edge-tts")

    async with semaphore:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
                await communicate.save(str(output_path))
                logger.debug(f"TTS done: segment {idx}")
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"TTS segment {idx} attempt {attempt+1} failed: {e}. Retry in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    logger.warning(f"TTS segment {idx} permanently failed after {max_retries} retries: {e}")
                    return False
    return False


async def _generate_all_tts(
    segments: list,
    tts_dir: Path,
    voice: str,
    rate: str,
    progress_callback: Optional[Callable] = None,
    total: int = 0,
) -> None:
    """Run all TTS tasks concurrently with a concurrency semaphore."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    completed = 0

    tasks = []
    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue
        raw_path = tts_dir / f"raw_{i:04d}.mp3"
        tasks.append((i, _tts_one_segment(semaphore, text, raw_path, voice, rate, i)))

    async def _tracked(idx, coro):
        nonlocal completed
        result = await coro
        completed += 1
        if progress_callback:
            progress_callback(
                0.60 + 0.20 * (completed / max(total, 1)),
                f"TTS dubbing {completed}/{total}...",
            )
        return result

    await asyncio.gather(*[_tracked(idx, coro) for idx, coro in tasks])


def generate_dubbed_audio(
    segments: list,
    video_duration: float,
    temp_dir: Path,
    output_path: Path,
    voice_id: str,
    progress_callback: Optional[Callable] = None,
) -> Path:
    """
    1. Generate TTS for all segments in PARALLEL (async + semaphore).
    2. Adjust speed for each segment to fit its original time slot.
    3. Overlay all clips onto a blank timeline.
    4. Export to MP3.
    """
    tts_dir = temp_dir / "dubbing_tts"
    ensure_dir(tts_dir)

    voice = voice_id
    rate = config.tts.rate
    total = sum(1 for s in segments if s.text.strip())

    logger.info(f"Starting parallel TTS for {total} segments (max {MAX_CONCURRENCY} concurrent)...")

    # --- Phase 1: Parallel TTS generation ---
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run,
                    _generate_all_tts(segments, tts_dir, voice, rate, progress_callback, total),
                )
                future.result()
        else:
            loop.run_until_complete(
                _generate_all_tts(segments, tts_dir, voice, rate, progress_callback, total)
            )
    except RuntimeError:
        asyncio.run(
            _generate_all_tts(segments, tts_dir, voice, rate, progress_callback, total)
        )

    # --- Phase 2: Build timeline ---
    logger.info("Building audio timeline...")
    if progress_callback:
        progress_callback(0.82, "Building audio timeline...")

    timeline = AudioSegment.silent(duration=int(video_duration * 1000))

    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue

        raw_path = tts_dir / f"raw_{i:04d}.mp3"
        fitted_path = tts_dir / f"fitted_{i:04d}.mp3"

        if not raw_path.exists() or raw_path.stat().st_size == 0:
            logger.warning(f"Segment {i} TTS file missing, skipping.")
            continue

        try:
            tts_dur = get_audio_duration(raw_path)
            target_dur = seg.end - seg.start

            if tts_dur > 0 and target_dur > 0 and tts_dur > target_dur * 1.05:
                ratio = tts_dur / target_dur
                ratio = min(ratio, 4.0)  # never speed up more than 4x
                logger.debug(f"Segment {i}: speed ratio {ratio:.2f}x")
                if not fitted_path.exists() or fitted_path.stat().st_size == 0:
                    adjust_audio_speed(raw_path, fitted_path, ratio)
                final_path = fitted_path
            else:
                final_path = raw_path

            clip = AudioSegment.from_file(str(final_path))
            start_ms = int(seg.start * 1000)
            timeline = timeline.overlay(clip, position=start_ms)

        except Exception as e:
            logger.warning(f"Segment {i} overlay failed: {e}")

    # --- Phase 3: Export ---
    logger.info("Exporting combined dubbing audio...")
    if progress_callback:
        progress_callback(0.85, "Exporting dubbed audio...")

    ensure_dir(output_path.parent)
    timeline.export(str(output_path), format="mp3", bitrate="192k")
    logger.info(f"Dubbed audio saved: {output_path}")

    return output_path
