"""
Main pipeline orchestrator.
Ties together all modules to execute the full video processing pipeline.
"""

import shutil
from pathlib import Path
from datetime import datetime
import time
import time
from typing import Optional, Callable

from app.downloader import download_video, is_valid_youtube_url, VideoInfo
from app.transcription import (
    extract_audio,
    transcribe_audio,
    save_transcript,
    transcript_to_text,
)
from app.llm import generate_script, save_script, load_script
from app.clipper import (
    match_clips,
    process_clips,
    add_crossfade_transition,
    get_video_duration,
)
from app.voice import generate_voiceover, get_audio_duration
from app.subtitle import text_to_subtitle_lines, generate_ass_subtitle, burn_subtitles
from app.render import (
    mix_audio_tracks,
    final_render,
    select_background_music,
)
from app.utils import get_logger, ensure_dir, reset_pipeline_workspace, safe_filename
from app.config import config

logger = get_logger(__name__)


class PipelineError(Exception):
    """Raised when a pipeline step fails."""
    pass


def run_pipeline(
    youtube_url: str,
    output_dir: Optional[Path] = None,
    language: str = "vi",
    voice_id: str = "vi-VN-HoaiMyNeural",
    trim_start: float = 0.0,
    trim_end: float = 0.0,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    keep_temp: bool = False,
) -> Path:
    """
    Execute the full movie-to-TikTok pipeline.

    Steps:
        1. Validate URL
        2. Download video
        3. Extract audio + transcribe
        4. Generate AI script
        5. Match clips to narration
        6. Process clips (cut + vertical)
        7. Generate voiceover
        8. Concatenate clips
        9. Mix audio (voice + music)
        10. Generate & burn subtitles
        11. Final render

    Args:
        youtube_url: YouTube URL to process.
        output_dir: Directory for final output (defaults to config).
        language: Narration/transcription language ('en' or 'vi').
        progress_callback: Optional callback(percent: float, message: str).
        keep_temp: If True, don't clean temp files after completion.

    Returns:
        Path to the final rendered video.

    Raises:
        PipelineError: On any pipeline failure.
        ValueError: On invalid URL.
    """
    def _cb(pct: float, msg: str):
        logger.info(f"[{pct*100:.0f}%] {msg}")
        if progress_callback:
            progress_callback(pct, msg)

    # Setup
    output_dir = output_dir or config.paths.output_dir
    ensure_dir(output_dir)
    temp_dir = config.paths.temp_dir

    # --- STEP 1: Validate URL ---
    _cb(0.0, "Validating URL...")
    if not is_valid_youtube_url(youtube_url):
        raise ValueError(f"Invalid YouTube URL: {youtube_url}")

    # Full wipe: every run starts fresh (no transcript/TTS/download from previous URL).
    _cb(0.01, "Xóa dữ liệu tạm video trước...")
    reset_pipeline_workspace(temp_dir, youtube_url, trim_start, trim_end)

    # --- STEP 2: Download Video ---
    try:
        _cb(0.02, "Downloading video from YouTube...")
        video_info: VideoInfo = download_video(
            youtube_url,
            output_dir=temp_dir / "downloads",
            progress_callback=progress_callback,
        )
        start_time = time.time()
    except Exception as e:
        raise PipelineError(f"Download failed: {e}") from e

    video_path = video_info.video_path
    movie_title = video_info.title

    # --- STEP 2.5: Trim Video ---
    if trim_end > trim_start >= 0 and trim_end > 0:
        _cb(0.15, f"Trimming video from {trim_start}s to {trim_end}s...")
        trimmed_path = temp_dir / "downloads" / f"trimmed_{int(trim_start)}_{int(trim_end)}_{video_path.name}"
        import subprocess
        from app.utils import find_ffmpeg
        cmd = [
            find_ffmpeg(), "-y",
            "-i", str(video_path),
            "-ss", str(trim_start),
            "-to", str(trim_end),
            "-c", "copy",
            str(trimmed_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise PipelineError(f"Trimming failed: {res.stderr}")
        video_path = trimmed_path

    audio_path: Optional[Path] = temp_dir / "audio.wav"
    step_times: dict[str, float] = {}

    # --- STEP 3: Transcribe Audio ---
    try:
        t_step = time.time()
        transcript_json_path = temp_dir / "transcript.json"
        _cb(0.33, "Extracting audio...")
        audio_path = extract_audio(video_path, temp_dir / "audio.wav")

        _cb(0.38, "Transcribing audio with Whisper...")
        transcript_segments = transcribe_audio(
            audio_path,
            language=None,
            progress_callback=progress_callback,
        )
        save_transcript(transcript_segments, transcript_json_path)
        step_times["transcribe"] = time.time() - t_step
        logger.info(f"Transcription completed in {step_times['transcribe']:.1f}s")

    except Exception as e:
        raise PipelineError(f"Transcription failed: {e}") from e

    transcript_text = transcript_to_text(transcript_segments)
    logger.info(f"Transcript length: {len(transcript_text)} chars")

    from app.transcription.segment_consolidator import consolidate_segments

    raw_count = len(transcript_segments)
    transcript_segments = consolidate_segments(transcript_segments)
    logger.info(
        f"Consolidated transcript: {raw_count} → {len(transcript_segments)} segments "
        "(fewer API calls for translate + dubbing)"
    )
    save_transcript(transcript_segments, transcript_json_path)

    # --- STEP 4: Translate ALL segments to Vietnamese (unconditional) ---
    # Rule: every input video in any language → 100% Vietnamese output.
    # We do NOT trust Whisper's detected language to decide whether to translate.
    # The only exception is if the video is already in Vietnamese (detected by translator).
    _cb(0.50, "Đang dịch sang tiếng Việt (100%)...")
    t_step = time.time()
    logger.info(f"[TRANSLATE] Forcing Vietnamese translation for ALL {len(transcript_segments)} segments...")
    from app.llm.translator import batch_translate_segments, TranslationError
    try:
        translated_segments = batch_translate_segments(
            transcript_segments,
            progress_callback=progress_callback,
        )
    except TranslationError as e:
        raise PipelineError(str(e)) from e
    step_times["translate"] = time.time() - t_step
    logger.info(f"[TRANSLATE] Done in {step_times['translate']:.1f}s")

    # --- STEP 5: Generate Dubbed Audio ---
    _cb(0.60, "Generating dubbed audio...")
    from app.voice.dubbing_mixer import generate_dubbed_audio
    from app.clipper import get_video_duration

    t_step = time.time()
    video_duration = get_video_duration(video_path)
    voice_path = temp_dir / "dubbed_voiceover.mp3"

    try:
        generate_dubbed_audio(
            segments=translated_segments,
            video_duration=video_duration,
            temp_dir=temp_dir,
            output_path=voice_path,
            voice_id=voice_id,
            progress_callback=progress_callback,
        )
    except Exception as e:
        raise PipelineError(f"Dubbing generation failed: {e}") from e
    step_times["dubbing"] = time.time() - t_step
    logger.info(f"Dubbing completed in {step_times['dubbing']:.1f}s (voice={voice_id})")

    # --- STEP 6: Mix Audio (Voice + Music over Original Video) ---
    try:
        t_step = time.time()
        _cb(0.75, "Mixing audio tracks...")
        music_path = select_background_music(config.paths.music_dir)
        audio_mixed_path = temp_dir / "with_audio.mp4"

        mix_audio_tracks(
            video_path=video_path,
            voice_path=voice_path,
            music_path=music_path,
            output_path=audio_mixed_path,
            voice_volume=config.music.voice_volume,
            music_volume=config.music.music_volume,
            fade_duration=config.music.fade_duration,
            progress_callback=progress_callback,
        )
        step_times["mix_audio"] = time.time() - t_step
        logger.info(f"Audio mix completed in {step_times['mix_audio']:.1f}s")
    except Exception as e:
        raise PipelineError(f"Audio mixing failed: {e}") from e

    # --- STEP 7: Generate & Burn Subtitles ---
    try:
        t_step = time.time()
        _cb(0.85, "Generating synced subtitles...")
        from app.subtitle.subtitle_generator import segments_to_subtitle_lines
        
        sub_lines = segments_to_subtitle_lines(
            segments=translated_segments,
            words_per_line=config.subtitle.words_per_line,
        )

        ass_path = temp_dir / "subtitles.ass"
        generate_ass_subtitle(
            subtitle_lines=sub_lines,
            output_path=ass_path,
            font_size=config.subtitle.font_size,
        )

        subbed_path = temp_dir / "with_subtitles.mp4"
        burn_subtitles(
            video_path=audio_mixed_path,
            subtitle_path=ass_path,
            output_path=subbed_path,
            progress_callback=progress_callback,
        )

        step_times["subtitles"] = time.time() - t_step
        logger.info(f"Subtitles completed in {step_times['subtitles']:.1f}s")
    except Exception as e:
        logger.warning(f"Subtitle generation failed: {e}. Skipping subtitles.")
        subbed_path = audio_mixed_path

    # --- STEP 10: Final Render ---
    try:
        t_step = time.time()
        _cb(0.93, "Final render...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_title = safe_filename(movie_title)[:40]
        output_filename = f"{clean_title}_{timestamp}.mp4"
        final_path = output_dir / output_filename

        final_render(
            video_path=subbed_path,
            output_path=final_path,
            target_fps=config.video.fps,
            width=config.video.width,
            height=config.video.height,
            use_gpu=config.video.use_gpu,
            progress_callback=progress_callback,
        )

        step_times["final_render"] = time.time() - t_step
        logger.info(f"Final render completed in {step_times['final_render']:.1f}s")
    except Exception as e:
        raise PipelineError(f"Final render failed: {e}") from e

    # Cleanup temp files
    if not keep_temp:
        try:
            for f in [voice_path, audio_mixed_path, subbed_path]:
                if f and Path(f).exists():
                    f.unlink()
            if audio_path and audio_path.exists():
                audio_path.unlink()
        except Exception:
            pass

    _cb(1.0, f"✅ Done! Output: {final_path.name}")
    logger.info(f"Pipeline complete. Output: {final_path}")
    if step_times:
        breakdown = ", ".join(f"{k}={v:.1f}s" for k, v in step_times.items())
        logger.info(f"Step timings: {breakdown}")

    # Save basic metadata
    _save_metadata(movie_title, youtube_url, output_dir, timestamp)

    return final_path


def _save_metadata(title: str, url: str, output_dir: Path, timestamp: str) -> None:
    """Save basic video metadata."""
    try:
        meta_path = output_dir / f"metadata_{timestamp}.txt"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"Source: {url}\n")
            f.write(f"Title: {title}\n\n")
            f.write(f"Dubbing processing completed on {datetime.now().isoformat()}\n")
        logger.info(f"Metadata saved: {meta_path}")
    except Exception as e:
        logger.warning(f"Could not save metadata: {e}")
