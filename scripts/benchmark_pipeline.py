"""Run pipeline with step timing (writes to logs/)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Short clip for benchmark (~19s)
URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


def main():
    from app.pipeline import run_pipeline, PipelineError

    print("Starting benchmark pipeline...", flush=True)
    t0 = time.time()
    try:
        out = run_pipeline(
            youtube_url=URL,
            language="vi",
            voice_id="vi-VN-HoaiMyNeural",
            trim_start=0.0,
            trim_end=0.0,
            progress_callback=lambda p, m: print(f"  [{p*100:5.1f}%] {m}", flush=True),
            keep_temp=True,
        )
        print(f"\nOK in {time.time()-t0:.1f}s -> {out}", flush=True)
    except PipelineError as e:
        print(f"\nFAILED in {time.time()-t0:.1f}s: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
