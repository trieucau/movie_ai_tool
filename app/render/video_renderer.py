"""
Final video render module.
Composites video clips, voiceover, background music, and subtitles
into the final TikTok-ready vertical video.
"""

import os
import subprocess
import random
from pathlib import Path
from typing import Optional, Callable, List

from app.utils import get_logger, ensure_dir, find_ffmpeg
from app.config import config

logger = get_logger(__name__)


def mix_audio_tracks(
    video_path: Path,
    voice_path: Path,
    music_path: Optional[Path],
    output_path: Path,
    voice_volume: float = 1.0,
    music_volume: float = 0.15,
    fade_duration: float = 2.0,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """
    Mix voiceover + background music with the video (replacing original audio).

    Args:
        video_path: Input video (with or without audio).
        voice_path: Voiceover audio path (MP3/WAV).
        music_path: Background music path (optional).
        output_path: Output video with mixed audio.
        voice_volume: Voiceover volume multiplier.
        music_volume: Background music volume multiplier.
        fade_duration: Fade in/out duration in seconds.
        progress_callback: Optional progress callback.

    Returns:
        Path to output video.
    """
    ensure_dir(output_path.parent)

    logger.info("Mixing audio tracks...")

    if progress_callback:
        progress_callback(0.83, "Mixing audio...")

    if music_path and music_path.exists():
        # Mix voice + music
        filter_complex = (
            f"[1:a]volume={voice_volume}[voice];"
            f"[2:a]volume={music_volume},afade=t=in:st=0:d={fade_duration},"
            f"afade=t=out:st=0:d={fade_duration}[music];"
            f"[voice][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )

        cmd = [
            find_ffmpeg(), "-y",
            "-i", str(video_path),
            "-i", str(voice_path),
            "-i", str(music_path),
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", config.video.audio_bitrate,
            str(output_path),
        ]
    else:
        # Voice only
        cmd = [
            find_ffmpeg(), "-y",
            "-i", str(video_path),
            "-i", str(voice_path),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", config.video.audio_bitrate,
            str(output_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio mixing failed: {result.stderr[-500:]}")

    logger.info(f"Audio mixed: {output_path}")
    return output_path


def final_render(
    video_path: Path,
    output_path: Path,
    target_fps: int = 60,
    width: int = 1080,
    height: int = 1920,
    use_gpu: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """
    Final render pass: upscale/normalize the video to exact TikTok specs.

    Args:
        video_path: Input video (already composed).
        output_path: Final output path.
        target_fps: Target frame rate.
        width: Output width.
        height: Output height.
        use_gpu: Use NVIDIA NVENC if available.
        progress_callback: Optional progress callback.

    Returns:
        Path to final rendered video.
    """
    ensure_dir(output_path.parent)

    skip_reencode = os.getenv("FINAL_RENDER_FAST", "true").lower() == "true"
    logger.info(f"Final render: {width}x{height}@{target_fps}fps (fast_copy={skip_reencode})")

    if progress_callback:
        progress_callback(0.93, "Final render...")

    if skip_reencode:
        cmd = [
            find_ffmpeg(), "-y",
            "-i", str(video_path),
            "-c", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]
    else:
        if use_gpu:
            video_codec = ["h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "20"]
        else:
            video_codec = ["libx264", "-preset", config.video.preset, "-crf", str(config.video.crf)]
        cmd = [
            find_ffmpeg(), "-y",
            "-i", str(video_path),
            "-r", str(target_fps),
            "-c:v", *video_codec,
            "-c:a", "aac",
            "-b:a", config.video.audio_bitrate,
            "-b:v", config.video.video_bitrate,
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Final render failed: {result.stderr[-500:]}")

    logger.info(f"Final video: {output_path}")

    if progress_callback:
        progress_callback(1.0, "Done!")

    return output_path


def select_background_music(music_dir: Path) -> Optional[Path]:
    """
    Select a random background music track from the music directory.

    Args:
        music_dir: Directory containing music files.

    Returns:
        Path to a music file, or None if directory is empty.
    """
    if not music_dir.exists():
        return None

    music_files = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))
    if not music_files:
        logger.info("No background music found in assets/music/")
        return None

    selected = random.choice(music_files)
    logger.info(f"Selected background music: {selected.name}")
    return selected


def trim_to_duration(
    video_path: Path,
    output_path: Path,
    max_duration: float,
) -> Path:
    """
    Trim video to a maximum duration.

    Args:
        video_path: Input video.
        output_path: Output path.
        max_duration: Maximum duration in seconds.

    Returns:
        Path to trimmed video.
    """
    cmd = [
        find_ffmpeg(), "-y",
        "-i", str(video_path),
        "-t", str(max_duration),
        "-c", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Trim failed: {result.stderr}")
    return output_path
