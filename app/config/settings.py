"""
Configuration management for Movie AI Tool.
Loads settings from .env and provides typed config objects.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load .env from project root
_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env")


@dataclass
class APIConfig:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    openai_base_url: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", None))
    elevenlabs_api_key: str = field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    elevenlabs_voice_id: str = field(default_factory=lambda: os.getenv("ELEVENLABS_VOICE_ID", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_whisper_model: str = field(default_factory=lambda: os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo"))



@dataclass
class PathConfig:
    root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def temp_dir(self) -> Path:
        return self.root / "temp"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def music_dir(self) -> Path:
        return self.assets_dir / "music"

    @property
    def fonts_dir(self) -> Path:
        return self.assets_dir / "fonts"


@dataclass
class VideoConfig:
    width: int = 1080
    height: int = 1920
    fps: int = int(os.getenv("OUTPUT_FPS", "30"))
    use_gpu: bool = field(default_factory=lambda: os.getenv("USE_GPU", "false").lower() == "true")
    # Dynamically select codec based on USE_GPU
    codec: str = field(default_factory=lambda: "h264_nvenc" if os.getenv("USE_GPU", "false").lower() == "true" else "libx264")
    crf: int = 18
    preset: str = "fast"
    audio_bitrate: str = "192k"
    video_bitrate: str = "8M"


@dataclass
class WhisperConfig:
    model_size: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))
    device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cpu"))
    compute_type: str = field(default_factory=lambda: os.getenv("WHISPER_COMPUTE", "int8"))
    language: Optional[str] = None  # None = auto-detect


@dataclass
class TTSConfig:
    engine: str = field(default_factory=lambda: os.getenv("TTS_ENGINE", "edge"))  # edge | elevenlabs
    edge_voice: str = field(default_factory=lambda: os.getenv("EDGE_VOICE", "vi-VN-HoaiMyNeural"))
    rate: str = field(default_factory=lambda: os.getenv("EDGE_TTS_RATE", "+0%"))
    volume: str = "+0%"


@dataclass
class SubtitleConfig:
    font_size: int = 72
    font_color: str = "white"
    stroke_color: str = "black"
    stroke_width: int = 3
    position: str = "center"
    font_name: str = "Arial-Bold"
    highlight_color: str = "#FFD700"
    words_per_line: int = 4


@dataclass
class MusicConfig:
    voice_volume: float = 1.0
    music_volume: float = 0.15
    fade_duration: float = 2.0


@dataclass
class AppConfig:
    api: APIConfig = field(default_factory=APIConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    music: MusicConfig = field(default_factory=MusicConfig)
    max_video_duration: int = int(os.getenv("MAX_VIDEO_DURATION", "90"))  # seconds for final video
    clip_min_duration: float = 2.0
    clip_max_duration: float = 8.0


# Global config singleton
config = AppConfig()
