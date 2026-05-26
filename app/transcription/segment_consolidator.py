"""
Merge fine-grained Whisper segments into fewer dubbing/translation units.
Cuts API calls (translate + TTS) without losing timestamps on the merged span.
"""

from typing import List

from app.transcription.whisper_transcriber import TranscriptSegment


def consolidate_segments(
    segments: List[TranscriptSegment],
    max_gap_s: float = 0.55,
    max_chars: int = 480,
) -> List[TranscriptSegment]:
    """
    Merge adjacent segments when pause is small and combined text is not too long.

    A 19s clip often yields 50–120 micro-segments from VAD; merging to ~5–15
    sentence-level units speeds translation and dubbing dramatically.
    """
    if not segments:
        return []

    merged: List[TranscriptSegment] = []
    buf = segments[0]
    buf_words = list(buf.words or [])

    for seg in segments[1:]:
        gap = seg.start - buf.end
        combined_len = len(buf.text) + 1 + len(seg.text)

        if gap <= max_gap_s and combined_len <= max_chars:
            buf = TranscriptSegment(
                start=buf.start,
                end=seg.end,
                text=f"{buf.text} {seg.text}".strip(),
                words=buf_words + list(seg.words or []),
            )
        else:
            merged.append(buf)
            buf = seg
            buf_words = list(seg.words or [])

    merged.append(buf)
    return merged
