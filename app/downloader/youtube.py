"""
YouTube video downloader using yt-dlp.
Downloads high-quality video for processing.
"""

import re
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

from app.utils import get_logger, ensure_dir, url_to_id, find_ffmpeg
from app.config import config

logger = get_logger(__name__)


import shutil

def _get_ytdlp_executable() -> str:
    """Return the full path to yt-dlp executable.

    Checks the same directory as the Python executable, the adjacent Scripts folder,
    and finally falls back to searching the system PATH.
    """
    scripts_dir = Path(sys.executable).parent
    scripts_folder = scripts_dir / "Scripts"
    for folder in (scripts_dir, scripts_folder):
        for name in ("yt-dlp.exe", "yt-dlp"):
            candidate = folder / name
            if candidate.exists():
                return str(candidate)
    which_path = shutil.which("yt-dlp")
    if which_path:
        return which_path
    return "yt-dlp"


@dataclass
class VideoInfo:
    url: str
    title: str
    duration: int  # seconds
    video_path: Path
    thumbnail_url: Optional[str] = None
    description: Optional[str] = None


def is_valid_youtube_url(url: str) -> bool:
    """Validate YouTube URL format."""
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+",
        r"(?:https?://)?youtu\.be/[\w-]+",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/[\w-]+",
    ]
    return any(re.match(p, url.strip()) for p in patterns)


def download_video(
    url: str,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> VideoInfo:
    """
    Download a YouTube video using yt-dlp.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the downloaded video.
        progress_callback: Optional callback(percent, status_msg).

    Returns:
        VideoInfo dataclass with path and metadata.

    Raises:
        ValueError: If URL is invalid.
        RuntimeError: If download fails.
    """
    if not is_valid_youtube_url(url):
        raise ValueError(f"Invalid YouTube URL: {url}")

    output_dir = output_dir or config.paths.temp_dir / "downloads"
    ensure_dir(output_dir)

    video_id = url_to_id(url)
    output_template = str(output_dir / f"{video_id}_%(id)s.%(ext)s")

    logger.info(f"Starting download: {url}")

    # Get video info first
    info = _get_video_info(url)
    title = info.get("title", "unknown")
    duration = info.get("duration", 0)
    thumbnail = info.get("thumbnail")

    logger.info(f"Video: '{title}' | Duration: {duration}s")

    if progress_callback:
        progress_callback(0.0, f"Downloading: {title}")

    # Build yt-dlp command
    ytdlp = _get_ytdlp_executable()
    cmd = [
        ytdlp,
        "--ffmpeg-location", find_ffmpeg(),
        "--format", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", output_template,
        "--no-playlist",
        "--write-info-json",
        "--progress",
        "--newline",
        url,
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    downloaded_path: Optional[Path] = None

    for line in process.stdout:
        line = line.strip()
        if not line:
            continue

        logger.debug(f"yt-dlp: {line}")

        # Parse progress percentage
        if "[download]" in line and "%" in line:
            try:
                pct_str = line.split("%")[0].split()[-1]
                pct = float(pct_str)
                if progress_callback:
                    progress_callback(pct / 100.0 * 0.3, f"Downloading... {pct:.0f}%")
            except (ValueError, IndexError):
                pass

        # Detect output filename
        if "[Merger]" in line or "Destination:" in line:
            parts = line.split("Destination:")
            if len(parts) > 1:
                downloaded_path = Path(parts[1].strip())

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"yt-dlp failed with code {process.returncode}")

    # Find the downloaded mp4 if not detected from output
    if downloaded_path is None or not downloaded_path.exists():
        mp4_files = sorted(output_dir.glob(f"{video_id}_*.mp4"), key=lambda p: p.stat().st_mtime)
        if not mp4_files:
            raise RuntimeError("Downloaded file not found after yt-dlp completed.")
        downloaded_path = mp4_files[-1]

    logger.info(f"Download complete: {downloaded_path}")

    if progress_callback:
        progress_callback(0.3, "Download complete.")

    return VideoInfo(
        url=url,
        title=title,
        duration=duration,
        video_path=downloaded_path,
        thumbnail_url=thumbnail,
        description=info.get("description", ""),
    )


def _get_video_info(url: str) -> dict:
    """
    Fetch video metadata using yt-dlp --dump-json.

    Args:
        url: YouTube URL.

    Returns:
        Dictionary of video metadata.
    """
    ytdlp = _get_ytdlp_executable()
    cmd = [ytdlp, "--dump-json", "--no-playlist", url]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        logger.warning(f"Could not fetch video info: {result.stderr}")
        return {}

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
