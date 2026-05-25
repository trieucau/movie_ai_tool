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
    language: str = "en",
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

    # --- URL-based cache invalidation ---
    # If a different YouTube URL is used, wipe all cached pipeline files so
    # the pipeline runs fresh from the beginning.
    session_url_file = temp_dir / "session_url.txt"
    cached_url = session_url_file.read_text(encoding="utf-8").strip() if session_url_file.exists() else None
    if cached_url != youtube_url.strip():
        logger.info("New URL detected — clearing cached pipeline files.")
        for stale in [
            "transcript.json", "script.json", "audio.wav",
            "voice.mp3", "merged.mp4", "with_audio.mp4",
            "with_subtitles.mp4", "subtitles.ass",
        ]:
            stale_path = temp_dir / stale
            if stale_path.exists():
                try:
                    stale_path.unlink()
                except PermissionError as e:
                    logger.warning(f"Could not delete {stale_path}: {e}")
        # Also wipe clip folders
        import shutil as _shutil
        for clip_folder in ["clips", "vertical"]:
            folder = temp_dir / clip_folder
            if folder.exists():
                _shutil.rmtree(folder, ignore_errors=True)
        # Save the new URL as current session
        session_url_file.write_text(youtube_url.strip(), encoding="utf-8")
    else:
        logger.info("Same URL — using cached pipeline files where available.")

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
                language=language if language != "en" else None,
                progress_callback=progress_callback,
            )
            save_transcript(transcript_segments, transcript_json_path)
        logger.info(f"Transcription completed in {time.time() - start_time:.2f}s")
        start_time = time.time()

    except Exception as e:
        raise PipelineError(f"Transcription failed: {e}") from e

    transcript_text = transcript_to_text(transcript_segments)
    logger.info(f"Transcript length: {len(transcript_text)} chars")

    # --- STEP 4: Translate transcript to Vietnamese if needed ---
    if language != "vi":
        from app.llm.translator import translate_to_vietnamese
        start_time = time.time()
        transcript_text = translate_to_vietnamese(transcript_text)
        logger.info(f"Translation completed in {time.time() - start_time:.2f}s")
    else:
        logger.info("Source language is Vietnamese; no translation needed.")

    # --- STEP 4: Generate AI Script ---
    try:
        # Script generation timing and retry
        script_json_path = temp_dir / "script.json"
        max_retries = 3
        attempt = 0
        while attempt < max_retries:
            try:
                if script_json_path.exists():
                    _cb(0.56, "Loading cached AI script...")
                    script = load_script(script_json_path)
                    logger.info("Loaded script from cache.")
                    break
                else:
                    _cb(0.56, "Generating AI script...")
                    script = generate_script(
                        movie_title=movie_title,
                        transcript_text=transcript_text,
                        language=language,
                        progress_callback=progress_callback,
                    )
                    save_script(script, script_json_path)
                    break
            except RuntimeError as e:
                if "RateLimitError" in str(e) and attempt < max_retries - 1:
                    attempt += 1
                    wait = 2 ** attempt
                    logger.warning(f"Rate limit hit, retry {attempt}/{max_retries} after {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    raise PipelineError(f"Script generation failed: {e}") from e
        logger.info(f"Script generation completed in {time.time() - start_time:.2f}s")
        start_time = time.time()

    except Exception as e:
        raise PipelineError(f"Script generation failed: {e}") from e

    # --- STEP 5: Match Clips to Script ---
    try:
        _cb(0.63, "Matching scenes to narration...")
        video_duration = get_video_duration(video_path)
        clip_selections = match_clips(
            script_segments=script.segments,
            transcript=transcript_segments,
            video_duration=video_duration,
            target_duration=config.max_video_duration,
        )
        logger.info(f"Matched {len(clip_selections)} clips")

    except Exception as e:
        raise PipelineError(f"Scene matching failed: {e}") from e

    # --- STEP 6: Process Clips (cut + vertical) ---
    try:
        _cb(0.64, "Cutting and converting clips to vertical...")
        vertical_clips = process_clips(
            video_path=video_path,
            selections=clip_selections,
            temp_dir=temp_dir,
            progress_callback=progress_callback,
        )

        if not vertical_clips:
            raise PipelineError("No clips were successfully processed.")

        _cb(0.77, "Merging clips...")
        merged_path = temp_dir / "merged.mp4"
        add_crossfade_transition(
            clip_paths=vertical_clips,
            output_path=merged_path,
            progress_callback=progress_callback,
        )

    except Exception as e:
        raise PipelineError(f"Clip processing failed: {e}") from e

    # --- STEP 7: Generate Voiceover ---
    try:
        voice_cache = temp_dir / "voice.mp3"
        if voice_cache.exists():
            _cb(0.79, "Using cached voiceover...")
            logger.info("Loaded voiceover from cache.")
            voice_path = voice_cache
        else:
            _cb(0.79, "Generating AI voiceover...")
            voice_path = generate_voiceover(
                text=script.full_narration,
                output_path=voice_cache,
                progress_callback=progress_callback,
            )
    except Exception as e:
        raise PipelineError(f"Voiceover generation failed: {e}") from e

    # --- STEP 8: Mix Audio ---
    try:
        _cb(0.82, "Mixing audio tracks...")
        music_path = select_background_music(config.paths.music_dir)
        audio_mixed_path = temp_dir / "with_audio.mp4"

        mix_audio_tracks(
            video_path=merged_path,
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

    # --- STEP 9: Generate Subtitles ---
    try:
        _cb(0.88, "Generating subtitles...")
        voice_duration = get_audio_duration(voice_path)

        sub_lines = text_to_subtitle_lines(
            text=script.full_narration,
            audio_duration=voice_duration or config.max_video_duration,
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
            for f in [audio_path, merged_path, audio_mixed_path, subbed_path]:
                if f.exists():
                    f.unlink()
        except Exception:
            pass

    _cb(1.0, f"✅ Done! Output: {final_path.name}")
    logger.info(f"Pipeline complete. Output: {final_path}")

    # Save metadata
    _save_metadata(script, movie_title, youtube_url, output_dir, timestamp)

    return final_path


def _save_metadata(script, title: str, url: str, output_dir: Path, timestamp: str) -> None:
    """Save script metadata (caption, hashtags) alongside the video."""
    try:
        meta_path = output_dir / f"metadata_{timestamp}.txt"
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"Title: {script.title}\n")
            f.write(f"Source: {url}\n")
            f.write(f"Movie: {title}\n\n")
            f.write(f"Caption:\n{script.caption}\n\n")
            f.write(f"Hashtags:\n{' '.join(script.hashtags)}\n\n")
            f.write(f"Hook:\n{script.hook}\n\n")
            f.write(f"Full Narration:\n{script.full_narration}\n")
        logger.info(f"Metadata saved: {meta_path}")
    except Exception as e:
        logger.warning(f"Could not save metadata: {e}")
