"""
AI voiceover generation module.
Supports Edge TTS (free) and ElevenLabs (premium).
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Callable, Literal

from app.utils import get_logger, ensure_dir, find_ffprobe
from app.config import config

logger = get_logger(__name__)

TTS_ENGINE = Literal["edge", "elevenlabs"]


async def _generate_edge_tts(
    text: str,
    output_path: Path,
    voice: str,
    rate: str = "+10%",
) -> None:
    """Generate TTS audio using Microsoft Edge TTS (free)."""
    try:
        import edge_tts
    except ImportError:
        raise ImportError("edge-tts not installed. Run: pip install edge-tts")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
            await communicate.save(str(output_path))
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Edge TTS attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise


def generate_voiceover(
    text: str,
    output_path: Optional[Path] = None,
    engine: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Path:
    """
    Generate AI voiceover from text.

    Args:
        text: Narration text to convert to speech.
        output_path: Path for output audio file (MP3).
        engine: TTS engine ('edge' or 'elevenlabs'). Defaults to config.
        progress_callback: Optional progress callback.

    Returns:
        Path to the generated audio file.
    """
    output_path = output_path or config.paths.temp_dir / "voice.mp3"
    ensure_dir(output_path.parent)

    engine = engine or config.tts.engine

    logger.info(f"Generating voiceover with {engine} TTS...")

    if progress_callback:
        progress_callback(0.63, f"Generating voice ({engine})...")

    if engine == "elevenlabs" and config.api.elevenlabs_api_key:
        _generate_elevenlabs(text, output_path)
    else:
        # Default to Edge TTS
        asyncio.run(_generate_edge_tts(
            text=text,
            output_path=output_path,
            voice=config.tts.edge_voice,
            rate=config.tts.rate,
        ))

    logger.info(f"Voiceover saved: {output_path}")

    if progress_callback:
        progress_callback(0.65, "Voiceover generated.")

    return output_path


def _generate_elevenlabs(text: str, output_path: Path) -> None:
    """Generate TTS audio using ElevenLabs API."""
    try:
        from elevenlabs import ElevenLabs, VoiceSettings
    except ImportError:
        raise ImportError("elevenlabs not installed. Run: pip install elevenlabs")

    client = ElevenLabs(api_key=config.api.elevenlabs_api_key)
    voice_id = config.api.elevenlabs_voice_id or "21m00Tcm4TlvDq8ikWAM"  # Rachel default

    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_turbo_v2",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.3,
            use_speaker_boost=True,
        ),
    )

    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of an audio file using ffprobe."""
    cmd = [
        find_ffprobe(),
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def list_edge_voices() -> list:
    """List available Edge TTS voices."""
    try:
        import edge_tts
        voices = asyncio.run(edge_tts.list_voices())
        return voices
    except Exception as e:
        logger.warning(f"Could not list Edge TTS voices: {e}")
        return []
