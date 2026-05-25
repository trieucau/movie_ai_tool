"""
Auto scene matching module.
Maps narration segments to video timestamps using keyword and semantic matching.
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

from app.transcription import TranscriptSegment
from app.llm import ScriptSegment
from app.utils import get_logger
from app.config import config

logger = get_logger(__name__)


@dataclass
class ClipSelection:
    start: float
    end: float
    score: float
    segment_index: int
    narration_text: str


def match_clips(
    script_segments: List[ScriptSegment],
    transcript: List[TranscriptSegment],
    video_duration: float,
    target_duration: float = 75.0,
) -> List[ClipSelection]:
    """
    Match script narration segments to video clips using keyword + transcript matching.

    Args:
        script_segments: AI-generated script segments with keywords.
        transcript: Full video transcript with timestamps.
        video_duration: Total duration of the source video in seconds.
        target_duration: Target total duration for the final video.

    Returns:
        List of ClipSelection with timestamps for each segment.
    """
    selections: List[ClipSelection] = []
    used_ranges: List[Tuple[float, float]] = []

    # Distribute clips across the video timeline
    num_segments = len(script_segments)
    section_size = video_duration / max(num_segments, 1)

    for i, seg in enumerate(script_segments):
        # Primary: search transcript for keyword matches
        best = _find_best_transcript_match(
            keywords=seg.clip_keywords,
            narration=seg.text,
            transcript=transcript,
            search_start=i * section_size * 0.5,
            search_end=(i + 2) * section_size,
            used_ranges=used_ranges,
        )

        if best is None:
            # Fallback: use evenly distributed timestamps
            t_start = i * section_size + section_size * 0.1
            duration = min(seg.duration_hint, config.clip_max_duration)
            t_end = min(t_start + duration, video_duration - 1)
            best = (t_start, t_end, 0.1)

        start, end, score = best

        # Enforce clip duration limits
        clip_dur = end - start
        if clip_dur < config.clip_min_duration:
            end = start + config.clip_min_duration
        elif clip_dur > config.clip_max_duration:
            end = start + config.clip_max_duration

        end = min(end, video_duration - 0.5)

        selections.append(
            ClipSelection(
                start=round(start, 2),
                end=round(end, 2),
                score=score,
                segment_index=i,
                narration_text=seg.text,
            )
        )
        used_ranges.append((start, end))
        logger.debug(f"Segment {i}: [{start:.1f}s - {end:.1f}s] score={score:.2f} | {seg.text[:50]}...")

    return selections


def _find_best_transcript_match(
    keywords: List[str],
    narration: str,
    transcript: List[TranscriptSegment],
    search_start: float,
    search_end: float,
    used_ranges: List[Tuple[float, float]],
) -> Optional[Tuple[float, float, float]]:
    """
    Find the best matching transcript segment using keyword overlap.

    Returns:
        Tuple of (start, end, score) or None if no match found.
    """
    if not transcript:
        return None

    # Normalize keywords
    keywords_lower = [k.lower() for k in keywords]
    narration_words = set(re.findall(r"\b\w+\b", narration.lower()))

    best_score = 0.0
    best_start = None
    best_end = None

    for seg in transcript:
        # Skip segments outside search window
        if seg.end < search_start or seg.start > search_end:
            continue

        # Skip already used ranges (allow slight overlap)
        if _overlaps(seg.start, seg.end, used_ranges, tolerance=1.0):
            continue

        seg_words = set(re.findall(r"\b\w+\b", seg.text.lower()))

        # Score: keyword overlap + narration word overlap
        kw_overlap = sum(1 for kw in keywords_lower if kw in seg.text.lower())
        narr_overlap = len(seg_words & narration_words) / max(len(narration_words), 1)
        score = kw_overlap * 0.6 + narr_overlap * 0.4

        if score > best_score:
            best_score = score
            best_start = seg.start
            best_end = seg.end

    if best_start is not None:
        return (best_start, best_end, best_score)

    return None


def _overlaps(
    start: float,
    end: float,
    used_ranges: List[Tuple[float, float]],
    tolerance: float = 0.5,
) -> bool:
    """Check if a time range overlaps with any previously used ranges."""
    for u_start, u_end in used_ranges:
        if start < u_end - tolerance and end > u_start + tolerance:
            return True
    return False
