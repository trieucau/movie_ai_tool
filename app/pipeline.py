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
    load_transcript,
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
from app.utils import get_logger, ensure_dir, clean_temp, safe_filename
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
    ensure_dir(temp_dir)

    # --- Always clear old temp data before each run ---
    # Prevents stale transcripts/translations/TTS from polluting the new run.
    logger.info("Clearing temp files from previous run...")
    for stale in [
        "transcript.json", "script.json", "audio.wav",
        "voice.mp3", "merged.mp4", "with_audio.mp4",
        "with_subtitles.mp4", "subtitles.ass", "dubbed_voiceover.mp3",
    ]:
        stale_path = temp_dir / stale
        if stale_path.exists():
            try:
                stale_path.unlink()
                logger.debug(f"Deleted stale temp file: {stale}")
            except PermissionError as e:
                logger.warning(f"Could not delete {stale_path}: {e}")

    # Also clear TTS segment folder
    tts_dir = temp_dir / "dubbing_tts"
    if tts_dir.exists():
        import shutil
        shutil.rmtree(tts_dir, ignore_errors=True)
        logger.debug("Cleared dubbing_tts folder.")


    # --- STEP 1: Validate URL ---
    _cb(0.0, "Validating URL...")
    if not is_valid_youtube_url(youtube_url):
        raise ValueError(f"Invalid YouTube URL: {youtube_url}")

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
        if trimmed_path.exists():
            _cb(0.16, "Using cached trimmed video...")
            video_path = trimmed_path
        else:
            import subprocess
            from app.utils import find_ffmpeg
            cmd = [
                find_ffmpeg(), "-y",
                "-i", str(video_path),
                "-ss", str(trim_start),
                "-to", str(trim_end),
                "-c", "copy",
                str(trimmed_path)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise PipelineError(f"Trimming failed: {res.stderr}")
            video_path = trimmed_path

    # --- STEP 3: Transcribe Audio ---
    try:
        # Transcription timing
        transcript_json_path = temp_dir / "transcript.json"
        if transcript_json_path.exists():
            _cb(0.38, "Loading cached transcript...")
            transcript_segments = load_transcript(transcript_json_path)
            logger.info("Loaded transcript from cache.")
        else:
            _cb(0.33, "Extracting audio...")
            audio_path = extract_audio(video_path, temp_dir / "audio.wav")

            _cb(0.38, "Transcribing audio with Whisper...")
            transcript_segments = transcribe_audio(
                audio_path,
                language=None,  # Always auto-detect source language
                progress_callback=progress_callback,
            )
            save_transcript(transcript_segments, transcript_json_path)
        logger.info(f"Transcription completed in {time.time() - start_time:.2f}s")
        start_time = time.time()

    except Exception as e:
        raise PipelineError(f"Transcription failed: {e}") from e

    transcript_text = transcript_to_text(transcript_segments)
    logger.info(f"Transcript length: {len(transcript_text)} chars")

    # --- STEP 4: Translate ALL segments to Vietnamese (unconditional) ---
    # Rule: every input video in any language → 100% Vietnamese output.
    # We do NOT trust Whisper's detected language to decide whether to translate.
    # The only exception is if the video is already in Vietnamese (detected by translator).
    _cb(0.50, "Đang dịch sang tiếng Việt (100%)...")
    start_time = time.time()
    logger.info(f"[TRANSLATE] Forcing Vietnamese translation for ALL {len(transcript_segments)} segments...")
    from app.llm.translator import batch_translate_segments
    translated_segments = batch_translate_segments(
        transcript_segments,
        progress_callback=progress_callback,
    )
    logger.info(f"[TRANSLATE] Done in {time.time() - start_time:.2f}s")

    # --- STEP 5: Generate Dubbed Audio ---
    _cb(0.60, "Generating dubbed audio...")
    from app.voice.dubbing_mixer import generate_dubbed_audio
    from app.clipper import get_video_duration
    
    video_duration = get_video_duration(video_path)
    voice_path = temp_dir / "dubbed_voiceover.mp3"
    
    try:
        if voice_path.exists():
            _cb(0.65, "Using cached dubbing...")
        else:
            generate_dubbed_audio(
                segments=translated_segments,
                video_duration=video_duration,
                temp_dir=temp_dir,
                output_path=voice_path,
                voice_id=voice_id,
                progress_callback=progress_callback
            )
    except Exception as e:
        raise PipelineError(f"Dubbing generation failed: {e}") from e

    # --- STEP 6: Mix Audio (Voice + Music over Original Video) ---
    try:
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
    except Exception as e:
        raise PipelineError(f"Audio mixing failed: {e}") from e

    # --- STEP 7: Generate & Burn Subtitles ---
    try:
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

    except Exception as e:
        logger.warning(f"Subtitle generation failed: {e}. Skipping subtitles.")
        subbed_path = audio_mixed_path

    # --- STEP 10: Final Render ---
    try:
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

    except Exception as e:
        raise PipelineError(f"Final render failed: {e}") from e

    # Cleanup temp files
    if not keep_temp:
        try:
            # Keep downloads and transcript, clean intermediate files
            for f in [audio_path, voice_path, audio_mixed_path, subbed_path]:
                if f.exists():
                    f.unlink()
        except Exception:
            pass

    _cb(1.0, f"✅ Done! Output: {final_path.name}")
    logger.info(f"Pipeline complete. Output: {final_path}")

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
