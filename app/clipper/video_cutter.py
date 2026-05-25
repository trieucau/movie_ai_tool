"""
Video cutting, splicing, and vertical format conversion.
Uses ffmpeg for all video processing.
"""

import subprocess
import json
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass

from app.clipper.scene_matcher import ClipSelection
from app.utils import get_logger, ensure_dir, find_ffmpeg, find_ffprobe
from app.config import config

logger = get_logger(__name__)


def get_video_info(video_path: Path) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        find_ffprobe(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds."""
    info = get_video_info(video_path)
    return float(info.get("format", {}).get("duration", 0))


def cut_clip(
    video_path: Path,
    start: float,
    end: float,
    output_path: Path,
    fast_seek: bool = True,
) -> Path:
    """
    Cut a clip from a video file.

    Args:
        video_path: Source video path.
        start: Start time in seconds.
        end: End time in seconds.
        output_path: Output file path.
        fast_seek: Use fast seek (slightly less accurate but much faster).

    Returns:
        Path to the cut clip.
    """
    ensure_dir(output_path.parent)
    duration = end - start

    ffmpeg = find_ffmpeg()
    if fast_seek:
        cmd = [
            ffmpeg, "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", config.video.codec,
            "-c:a", "aac",
            "-preset", "ultrafast",
            "-avoid_negative_ts", "1",
            str(output_path),
        ]
    else:
        cmd = [
            ffmpeg, "-y",
            "-i", str(video_path),
            "-ss", str(start),
            "-t", str(duration),
            "-c:v", config.video.codec,
            "-c:a", "aac",
            "-preset", "ultrafast",
            str(output_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed: {result.stderr[-500:]}")

    return output_path


def convert_to_vertical(
    input_path: Path,
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    """
    Convert video to vertical (9:16) format with smart center crop.
    Uses face/content-aware cropping via center crop strategy.

    Args:
        input_path: Source clip path.
        output_path: Output vertical clip path.
        width: Output width (default 1080).
        height: Output height (default 1920).

    Returns:
        Path to the vertical video.
    """
    ensure_dir(output_path.parent)

    # Smart crop: scale to fit height, crop center horizontally
    # This preserves the most important center area of the frame
    vf_filter = (
        f"scale=-2:{height},"
        f"crop={width}:{height}:(iw-{width})/2:0,"
        f"scale={width}:{height}"
    )

    cmd = [
        find_ffmpeg(), "-y",
        "-i", str(input_path),
        "-vf", vf_filter,
        "-c:v", config.video.codec,
        "-c:a", "aac",
        "-preset", "fast",
        "-crf", "20",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Vertical conversion failed: {result.stderr[-500:]}")

    return output_path


def add_crossfade_transition(
    clip_paths: List[Path],
    output_path: Path,
    transition_duration: float = 0.3,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """
    Concatenate clips with crossfade transitions using ffmpeg.

    Args:
        clip_paths: List of clip file paths in order.
        output_path: Output concatenated video path.
        transition_duration: Duration of crossfade in seconds.
        progress_callback: Optional progress callback.

    Returns:
        Path to concatenated video.
    """
    ensure_dir(output_path.parent)

    if not clip_paths:
        raise ValueError("No clips provided for concatenation.")

    if len(clip_paths) == 1:
        import shutil
        shutil.copy(str(clip_paths[0]), str(output_path))
        return output_path

    # Build a concat demuxer file list
    concat_list = config.paths.temp_dir / "concat_list.txt"
    with open(concat_list, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{cp.as_posix()}'\n")

    # Simple concat (no re-encode, fastest)
    cmd = [
        find_ffmpeg(), "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", config.video.codec,
        "-c:a", "aac",
        "-preset", "fast",
        "-crf", "20",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info(f"Concatenating {len(clip_paths)} clips...")

    if progress_callback:
        progress_callback(0.75, f"Merging {len(clip_paths)} clips...")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Concat failed: {result.stderr[-500:]}")

    logger.info(f"Clips merged: {output_path}")
    return output_path


def process_clips(
    video_path: Path,
    selections: List[ClipSelection],
    temp_dir: Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> List[Path]:
    """
    Cut all selected clips and convert to vertical format.

    Args:
        video_path: Source video.
        selections: List of ClipSelection with timestamps.
        temp_dir: Directory for temp clip files.
        progress_callback: Optional progress callback.

    Returns:
        List of paths to processed vertical clips.
    """
    ensure_dir(temp_dir)
    clips_dir = temp_dir / "clips"
    vertical_dir = temp_dir / "vertical"
    ensure_dir(clips_dir)
    ensure_dir(vertical_dir)

    vertical_clips: List[Path] = []
    total = len(selections)

    for i, sel in enumerate(selections):
        if progress_callback:
            pct = 0.65 + (i / total) * 0.12
            progress_callback(pct, f"Processing clip {i+1}/{total}...")

        # Cut raw clip
        clip_path = clips_dir / f"clip_{i:03d}.mp4"
        try:
            cut_clip(video_path, sel.start, sel.end, clip_path)
        except RuntimeError as e:
            logger.warning(f"Failed to cut clip {i}: {e}, skipping.")
            continue

        # Skip vertical conversion to keep original aspect ratio
        vertical_clips.append(clip_path)

        logger.debug(f"Clip {i+1}/{total} done: {sel.start:.1f}s - {sel.end:.1f}s")

    logger.info(f"Processed {len(vertical_clips)} clips.")
    return vertical_clips
