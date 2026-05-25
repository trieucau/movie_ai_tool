"""
Subtitle generation and burn-in module.
Creates TikTok-style word-by-word karaoke subtitles burned into video.
"""

import subprocess
import json
import re
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass

from app.utils import get_logger, ensure_dir, find_ffmpeg
from app.config import config

logger = get_logger(__name__)


@dataclass
class SubtitleWord:
    start: float
    end: float
    word: str
    is_highlight: bool = False


@dataclass
class SubtitleLine:
    start: float
    end: float
    text: str
    words: List[SubtitleWord]


def text_to_subtitle_lines(
    text: str,
    audio_duration: float,
    words_per_line: int = 4,
) -> List[SubtitleLine]:
    """
    Convert narration text to evenly-timed subtitle lines.
    Used when word-level timestamps from TTS are not available.

    Args:
        text: Full narration text.
        audio_duration: Total audio duration in seconds.
        words_per_line: Max words per subtitle line.

    Returns:
        List of SubtitleLine with estimated timestamps.
    """
    words = text.split()
    if not words:
        return []

    time_per_word = audio_duration / len(words)
    lines: List[SubtitleLine] = []
    current_pos = 0.0

    for i in range(0, len(words), words_per_line):
        chunk = words[i: i + words_per_line]
        line_duration = len(chunk) * time_per_word
        line_end = current_pos + line_duration

        word_objs = []
        word_start = current_pos
        for w in chunk:
            word_end = word_start + time_per_word
            word_objs.append(SubtitleWord(
                start=round(word_start, 3),
                end=round(word_end, 3),
                word=w,
            ))
            word_start = word_end

        lines.append(SubtitleLine(
            start=round(current_pos, 3),
            end=round(line_end, 3),
            text=" ".join(chunk),
            words=word_objs,
        ))
        current_pos = line_end

    return lines


def segments_to_subtitle_lines(
    segments: list,
    words_per_line: int = 4,
) -> List[SubtitleLine]:
    """
    Convert a list of translated TranscriptSegments into SubtitleLine objects.
    Distributes translated words evenly within each segment's exact timestamps.
    """
    lines: List[SubtitleLine] = []
    
    for seg in segments:
        duration = seg.end - seg.start
        if duration <= 0 or not seg.text.strip():
            continue
            
        words = seg.text.split()
        if not words:
            continue
            
        time_per_word = duration / len(words)
        current_pos = seg.start
        
        for i in range(0, len(words), words_per_line):
            chunk = words[i: i + words_per_line]
            line_duration = len(chunk) * time_per_word
            line_end = current_pos + line_duration
            
            word_objs = []
            word_start = current_pos
            for w in chunk:
                word_end = word_start + time_per_word
                word_objs.append(SubtitleWord(
                    start=round(word_start, 3),
                    end=round(word_end, 3),
                    word=w,
                ))
                word_start = word_end
                
            lines.append(SubtitleLine(
                start=round(current_pos, 3),
                end=round(line_end, 3),
                text=" ".join(chunk),
                words=word_objs,
            ))
            current_pos = line_end
            
    return lines

def generate_ass_subtitle(
    subtitle_lines: List[SubtitleLine],
    output_path: Path,
    font_size: int = 72,
    font_name: str = "Arial Bold",
    primary_color: str = "&H00FFFFFF",   # white
    outline_color: str = "&H00000000",   # black
    highlight_color: str = "&H0000D7FF", # gold
    outline_width: int = 3,
    shadow: int = 1,
) -> Path:
    """
    Generate ASS subtitle file for TikTok-style karaoke display.

    Args:
        subtitle_lines: List of subtitle lines with word timestamps.
        output_path: Path to save the .ass file.
        font_size: Font size in points.
        font_name: Font family name.
        primary_color: Text color in ASS format (&HAABBGGRR).
        outline_color: Outline color in ASS format.
        highlight_color: Active word highlight color.
        outline_width: Outline thickness.
        shadow: Shadow depth.

    Returns:
        Path to generated ASS file.
    """
    ensure_dir(output_path.parent)

    # ASS file header
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},{highlight_color},{outline_color},&H80000000,-1,0,0,0,100,100,0,0,1,{outline_width},{shadow},2,20,20,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    for line in subtitle_lines:
        start_str = _format_ass_time(line.start)
        end_str = _format_ass_time(line.end)

        # Build karaoke text with word-by-word highlight timing
        karaoke_text = ""
        for word in line.words:
            dur_cs = int((word.end - word.start) * 100)  # centiseconds
            karaoke_text += f"{{\\k{dur_cs}}}{word.word} "
        karaoke_text = karaoke_text.strip()

        events.append(
            f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{karaoke_text}"
        )

    ass_content = header + "\n".join(events) + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    logger.info(f"Subtitle file saved: {output_path}")
    return output_path


def _format_ass_time(seconds: float) -> str:
    """Format seconds to ASS time format H:MM:SS.cc"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def burn_subtitles(
    video_path: Path,
    subtitle_path: Path,
    output_path: Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """
    Burn ASS subtitle file into video using ffmpeg.

    Args:
        video_path: Input video path.
        subtitle_path: Path to .ass subtitle file.
        output_path: Output video with burned subtitles.
        progress_callback: Optional progress callback.

    Returns:
        Path to output video with subtitles burned in.
    """
    ensure_dir(output_path.parent)

    logger.info("Burning subtitles into video...")

    if progress_callback:
        progress_callback(0.90, "Burning subtitles...")

    # Escape path for ffmpeg ass filter (Windows backslash issue)
    sub_path_escaped = str(subtitle_path).replace("\\", "/").replace(":", "\\:")

    cmd = [
        find_ffmpeg(), "-y",
        "-i", str(video_path),
        "-vf", f"ass='{sub_path_escaped}'",
        "-c:v", config.video.codec,
        "-c:a", "copy",
        "-preset", "fast",
        "-crf", str(config.video.crf),
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"Subtitle burn failed: {result.stderr[-300:]}")
        logger.warning("Falling back to subtitles without burn-in...")
        import shutil
        shutil.copy(str(video_path), str(output_path))

    logger.info(f"Subtitles burned: {output_path}")
    return output_path
