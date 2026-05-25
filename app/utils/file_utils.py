"""
File and path utility helpers.
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_temp(temp_dir: Path) -> None:
    """Remove all files in temp directory."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cleaned temp directory: {temp_dir}")


def url_to_id(url: str) -> str:
    """Generate a short hash ID from a URL for temp file naming."""
    return hashlib.md5(url.encode()).hexdigest()[:8]


def get_file_size_mb(path: Path) -> float:
    """Return file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)


def safe_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- ")
    return "".join(c if c in keep else "_" for c in name).strip()


def find_ffmpeg() -> Optional[str]:
    """Locate ffmpeg executable on PATH or common Windows paths."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    common = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\tools\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    ]
    for path in common:
        if os.path.isfile(path):
            return path
            
    # Check WinGet installation path
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        winget_path = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if winget_path.exists():
            import glob
            matches = glob.glob(str(winget_path / "Gyan.FFmpeg*" / "**" / "ffmpeg.exe"), recursive=True)
            if matches:
                return matches[0]

    return "ffmpeg"  # fallback, let OS raise the error


def find_ffprobe() -> Optional[str]:
    """Locate ffprobe executable on PATH or common Windows paths."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe

    common = [
        r"C:\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
        r"C:\tools\ffmpeg\bin\ffprobe.exe",
        r"C:\ProgramData\chocolatey\bin\ffprobe.exe",
    ]
    for path in common:
        if os.path.isfile(path):
            return path
            
    # Check WinGet installation path
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        winget_path = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if winget_path.exists():
            import glob
            matches = glob.glob(str(winget_path / "Gyan.FFmpeg*" / "**" / "ffprobe.exe"), recursive=True)
            if matches:
                return matches[0]

    return "ffprobe"  # fallback
