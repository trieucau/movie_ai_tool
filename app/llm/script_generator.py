"""
AI script generation module.
Uses OpenAI API to create viral TikTok-style movie review scripts.
"""

import json
import re
from typing import List, Optional, Callable
from dataclasses import dataclass, asdict, field

from app.utils import get_logger
from app.config import config

logger = get_logger(__name__)


@dataclass
class ScriptSegment:
    text: str
    emotion: str  # dramatic, suspense, curious, exciting, shocking
    clip_keywords: List[str] = field(default_factory=list)
    duration_hint: float = 5.0  # suggested duration in seconds


@dataclass
class MovieScript:
    title: str
    hook: str
    segments: List[ScriptSegment]
    caption: str
    hashtags: List[str]
    full_narration: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def total_text(self) -> str:
        return " ".join([self.hook] + [s.text for s in self.segments])


_SYSTEM_PROMPT = """You are a viral TikTok movie review content creator. 
You specialize in creating SHORT, PUNCHY, HIGH-RETENTION video scripts that hook viewers in the first 3 seconds.

Your style:
- Start with a shocking/dramatic hook
- Use short punchy sentences (5-10 words max per sentence)
- Create suspense and curiosity gaps
- Use emotional language: shocking, unbelievable, heartbreaking, twisted
- Fast pacing - new information every 3-5 seconds
- Keep total runtime under 90 seconds when spoken aloud
- NO spoilers of the very ending, tease but don't reveal

Output ONLY valid JSON, no markdown, no explanation."""

_USER_PROMPT_TEMPLATE = """Create a viral TikTok movie review script for this movie.

Movie Title: {title}
Movie Transcript Summary:
{transcript_summary}

Requirements:
- Hook: 1 shocking sentence (max 15 words) that grabs attention immediately
- 6-10 short dramatic segments (each 2-4 sentences, TikTok pacing)
- Each segment: specify emotion and 2-3 keywords describing the visual scene
- Caption: 1-2 sentences for post description  
- 5-8 relevant hashtags

Return ONLY this JSON structure:
{{
  "title": "Catchy video title",
  "hook": "The shocking opening line...",
  "segments": [
    {{
      "text": "Segment narration text here...",
      "emotion": "dramatic",
      "clip_keywords": ["keyword1", "keyword2"],
      "duration_hint": 5.0
    }}
  ],
  "caption": "Post caption text",
  "hashtags": ["#moviereview", "#tiktok"]
}}"""


def generate_script(
    movie_title: str,
    transcript_text: str,
    language: str = "en",
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> MovieScript:
    """
    Generate a viral TikTok movie review script using OpenAI.

    Args:
        movie_title: Title of the movie.
        transcript_text: Full transcript text from the video.
        language: Output language ('en' or 'vi').
        progress_callback: Optional progress callback.

    Returns:
        MovieScript dataclass.

    Raises:
        RuntimeError: If API call fails.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    if not config.api.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env file")

    client = OpenAI(
        api_key=config.api.openai_api_key,
        base_url=config.api.openai_base_url if config.api.openai_base_url else None
    )

    # Truncate transcript to avoid token limits (keep ~2000 chars)
    summary = transcript_text[:3000] if len(transcript_text) > 3000 else transcript_text

    lang_note = ""
    if language == "vi":
        lang_note = "\n\nIMPORTANT: Write the entire script in Vietnamese (Tiếng Việt)."

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        title=movie_title,
        transcript_summary=summary,
    ) + lang_note

    logger.info(f"Generating script for: {movie_title}")

    if progress_callback:
        progress_callback(0.57, "Generating AI script...")

    max_retries = 3
    import time
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=config.api.openai_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.85,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            break
        except Exception as e:
            if "429" in str(e) or "RateLimitError" in str(type(e).__name__):
                if attempt < max_retries - 1:
                    sleep_time = 10 * (attempt + 1)
                    logger.warning(f"Rate limited. Retrying in {sleep_time}s... ({e})")
                    if progress_callback:
                        progress_callback(0.57, f"Rate limited. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"OpenAI API rate limit exceeded: {e}")
            else:
                raise RuntimeError(f"OpenAI API error: {e}")

    raw = response.choices[0].message.content
    logger.debug(f"Raw AI response: {raw[:500]}...")

    script = _parse_script_response(raw)

    # Build full narration
    script.full_narration = script.hook + " " + " ".join(s.text for s in script.segments)

    logger.info(f"Script generated: {len(script.segments)} segments")

    if progress_callback:
        progress_callback(0.62, "Script generated!")

    return script


def _parse_script_response(raw: str) -> MovieScript:
    """Parse the JSON response from OpenAI into a MovieScript."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}\nRaw: {raw[:500]}")
        raise RuntimeError(f"Invalid JSON from AI: {e}")

    segments = []
    for seg_data in data.get("segments", []):
        segments.append(
            ScriptSegment(
                text=seg_data.get("text", ""),
                emotion=seg_data.get("emotion", "dramatic"),
                clip_keywords=seg_data.get("clip_keywords", []),
                duration_hint=float(seg_data.get("duration_hint", 5.0)),
            )
        )

    return MovieScript(
        title=data.get("title", "Movie Review"),
        hook=data.get("hook", ""),
        segments=segments,
        caption=data.get("caption", ""),
        hashtags=data.get("hashtags", ["#moviereview"]),
    )


def save_script(script: MovieScript, output_path) -> None:
    """Save script to JSON file."""
    from pathlib import Path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(script.to_dict(), f, ensure_ascii=False, indent=2)

    logger.info(f"Script saved: {output_path}")


def load_script(path) -> MovieScript:
    """Load script from a saved JSON file."""
    from pathlib import Path
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = [
        ScriptSegment(
            text=s["text"],
            emotion=s["emotion"],
            clip_keywords=s.get("clip_keywords", []),
            duration_hint=s.get("duration_hint", 5.0),
        )
        for s in data.get("segments", [])
    ]

    return MovieScript(
        title=data.get("title", ""),
        hook=data.get("hook", ""),
        segments=segments,
        caption=data.get("caption", ""),
        hashtags=data.get("hashtags", []),
        full_narration=data.get("full_narration", ""),
    )

