"""
Movie AI Tool - GUI Application
Dark, modern UI built with customtkinter.
"""

import sys
import threading
import queue
import traceback
from pathlib import Path
from typing import Optional

try:
    import customtkinter as ctk
    from tkinter import filedialog, messagebox
except ImportError:
    print("ERROR: customtkinter not installed. Run: pip install customtkinter")
    sys.exit(1)

from app.utils import get_logger

logger = get_logger(__name__)

# Theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

ACCENT = "#FF4757"
ACCENT2 = "#FFA502"
BG = "#0D0D0D"
SURFACE = "#1A1A2E"
SURFACE2 = "#16213E"
TEXT = "#FFFFFF"
SUBTEXT = "#A0A0B0"
SUCCESS = "#2ECC71"
ERROR = "#E74C3C"


class MovieAIApp(ctk.CTk):
    """Main application window for Movie AI Tool."""

    def __init__(self):
        super().__init__()

        self.title("🎬 Movie AI Tool - TikTok Review Generator")
        self.geometry("900x720")
        self.minsize(800, 640)
        self.configure(fg_color=BG)

        # State
        self._output_dir: Optional[Path] = None
        self._running = False
        self._log_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._poll_log_queue()

    # ─────────────────────────────────────────
    # UI CONSTRUCTION
    # ─────────────────────────────────────────

    def _build_ui(self):
        """Build all UI widgets."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._build_header()
        self._build_input_panel()
        self._build_log_panel()
        self._build_footer()

    def _build_header(self):
        """App header with title and subtitle."""
        header = ctk.CTkFrame(self, fg_color=SURFACE, corner_radius=0, height=80)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1)

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, pady=12, padx=20, sticky="w")

        ctk.CTkLabel(
            title_frame,
            text="🎬 Movie AI Tool",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color=ACCENT,
        ).pack(side="left", padx=(0, 12))

        ctk.CTkLabel(
            title_frame,
            text="YouTube → TikTok Review Generator",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=SUBTEXT,
        ).pack(side="left", pady=4)

    def _build_input_panel(self):
        """Input controls panel."""
        panel = ctk.CTkFrame(self, fg_color=SURFACE2, corner_radius=12)
        panel.grid(row=1, column=0, padx=20, pady=(16, 0), sticky="ew")
        panel.grid_columnconfigure(1, weight=1)

        # ── YouTube URL ──
        ctk.CTkLabel(
            panel, text="YouTube URL",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=SUBTEXT,
        ).grid(row=0, column=0, padx=(20, 10), pady=(20, 4), sticky="w")

        self._url_entry = ctk.CTkEntry(
            panel,
            placeholder_text="https://youtube.com/watch?v=...",
            font=ctk.CTkFont(size=14),
            fg_color="#0D0D0D",
            border_color=ACCENT,
            border_width=2,
            height=44,
            text_color=TEXT,
        )
        self._url_entry.grid(row=0, column=1, padx=(0, 10), pady=(20, 4), sticky="ew")

        self._load_info_btn = ctk.CTkButton(
            panel, text="Load Info", width=80, height=34,
            command=self._load_video_info,
            fg_color=SURFACE, hover_color="#2A2A4E",
            border_color=ACCENT, border_width=1, text_color=TEXT,
            font=ctk.CTkFont(size=13)
        )
        self._load_info_btn.grid(row=0, column=2, padx=(0, 20), pady=(20, 4))

        # ── Output Folder ──
        ctk.CTkLabel(
            panel, text="Output Folder",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=SUBTEXT,
        ).grid(row=1, column=0, padx=(20, 10), pady=(4, 4), sticky="w")

        self._output_label = ctk.CTkLabel(
            panel,
            text="(using default: output/)",
            font=ctk.CTkFont(size=13),
            text_color=SUBTEXT,
            anchor="w",
        )
        self._output_label.grid(row=1, column=1, padx=0, pady=(4, 4), sticky="ew")

        ctk.CTkButton(
            panel,
            text="Browse",
            width=100,
            height=34,
            fg_color=SURFACE,
            hover_color="#2A2A4E",
            border_color=ACCENT,
            border_width=1,
            text_color=TEXT,
            font=ctk.CTkFont(size=13),
            command=self._browse_output,
        ).grid(row=1, column=2, padx=(8, 20), pady=(4, 4))

        # ── Output language: always Vietnamese (fixed) ──
        ctk.CTkLabel(
            panel, text="Output Language",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=SUBTEXT,
        ).grid(row=2, column=0, padx=(20, 10), pady=(4, 4), sticky="w")

        ctk.CTkLabel(
            panel,
            text="🇻🇳  Tiếng Việt  (bắt buộc)",
            font=ctk.CTkFont(size=13),
            text_color="#00C896",
            anchor="w",
        ).grid(row=2, column=1, padx=0, pady=(4, 4), sticky="w")

        # language is always vi — no dropdown needed
        self._lang_var = ctk.StringVar(value="Vietnamese")

        # ── Voice selector ──
        ctk.CTkLabel(
            panel, text="Voice",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=SUBTEXT,
        ).grid(row=3, column=0, padx=(20, 10), pady=(4, 4), sticky="w")

        self._voice_var = ctk.StringVar(value="Tiếng Việt (Nữ - Hoài My)")
        voice_menu = ctk.CTkOptionMenu(
            panel,
            values=["Tiếng Việt (Nữ - Hoài My)", "Tiếng Việt (Nam - Nam Minh)"],
            variable=self._voice_var,
            fg_color=SURFACE,
            button_color=ACCENT,
            button_hover_color="#CC3344",
            dropdown_fg_color=SURFACE,
            text_color=TEXT,
            font=ctk.CTkFont(size=13),
            width=220,
        )
        voice_menu.grid(row=3, column=1, padx=0, pady=(4, 4), sticky="w")

        # ── Trim Video ──
        ctk.CTkLabel(
            panel, text="Trim Video",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=SUBTEXT,
        ).grid(row=4, column=0, padx=(20, 10), pady=(4, 16), sticky="w")

        trim_frame = ctk.CTkFrame(panel, fg_color="transparent")
        trim_frame.grid(row=4, column=1, columnspan=2, padx=0, pady=(4, 16), sticky="ew")
        trim_frame.grid_columnconfigure(0, weight=1)

        self._start_slider = ctk.CTkSlider(trim_frame, from_=0, to=100, command=self._on_start_slider)
        self._start_slider.set(0)
        self._start_slider.grid(row=0, column=0, padx=(0,10), pady=0, sticky="ew")
        self._start_label = ctk.CTkLabel(trim_frame, text="Start: 00:00", font=ctk.CTkFont(size=12), text_color=SUBTEXT, width=80, anchor="w")
        self._start_label.grid(row=0, column=1, padx=0, pady=0)

        self._end_slider = ctk.CTkSlider(trim_frame, from_=0, to=100, command=self._on_end_slider)
        self._end_slider.set(100)
        self._end_slider.grid(row=1, column=0, padx=(0,10), pady=(8,0), sticky="ew")
        self._end_label = ctk.CTkLabel(trim_frame, text="End: 00:00", font=ctk.CTkFont(size=12), text_color=SUBTEXT, width=80, anchor="w")
        self._end_label.grid(row=1, column=1, padx=0, pady=(8,0))

        self._video_duration = 0.0

        # ── Run Button ──
        self._run_btn = ctk.CTkButton(
            panel,
            text="▶  Generate TikTok Video",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            fg_color=ACCENT,
            hover_color="#CC3344",
            text_color=TEXT,
            corner_radius=10,
            command=self._start_pipeline,
        )
        self._run_btn.grid(row=5, column=0, columnspan=3, padx=20, pady=(0, 20), sticky="ew")

        # ── Progress Bar ──
        self._progress_bar = ctk.CTkProgressBar(
            panel,
            mode="determinate",
            progress_color=ACCENT2,
            fg_color="#1A1A2E",
            height=8,
        )
        self._progress_bar.set(0)
        self._progress_bar.grid(row=6, column=0, columnspan=3, padx=20, pady=(0, 6), sticky="ew")

        # ── Status Label ──
        self._status_label = ctk.CTkLabel(
            panel,
            text="Ready",
            font=ctk.CTkFont(size=12),
            text_color=SUBTEXT,
            anchor="w",
        )
        self._status_label.grid(row=7, column=0, columnspan=3, padx=20, pady=(0, 16), sticky="ew")

    def _build_log_panel(self):
        """Log output panel."""
        log_frame = ctk.CTkFrame(self, fg_color=SURFACE2, corner_radius=12)
        log_frame.grid(row=2, column=0, padx=20, pady=16, sticky="nsew")
        log_frame.grid_rowconfigure(1, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        # Header row
        header_row = ctk.CTkFrame(log_frame, fg_color="transparent")
        header_row.grid(row=0, column=0, padx=16, pady=(12, 4), sticky="ew")
        header_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header_row,
            text="📋 Pipeline Logs",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=SUBTEXT,
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkButton(
            header_row,
            text="Clear",
            width=70,
            height=28,
            fg_color=SURFACE,
            hover_color="#2A2A4E",
            text_color=SUBTEXT,
            font=ctk.CTkFont(size=12),
            command=self._clear_logs,
        ).grid(row=0, column=1, sticky="e")

        # Log textbox
        self._log_box = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(family="Consolas", size=12),
            fg_color="#0A0A14",
            text_color="#C8C8D8",
            wrap="word",
            scrollbar_button_color=SURFACE,
            scrollbar_button_hover_color=ACCENT,
            state="disabled",
        )
        self._log_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

    def _build_footer(self):
        """Footer with branding."""
        footer = ctk.CTkFrame(self, fg_color="transparent", height=28)
        footer.grid(row=3, column=0, sticky="ew")
        footer.grid_propagate(False)

        ctk.CTkLabel(
            footer,
            text="Movie AI Tool  •  Powered by faster-whisper + OpenAI + Edge TTS  •  ffmpeg",
            font=ctk.CTkFont(size=11),
            text_color="#444460",
        ).pack(pady=4)

    # ─────────────────────────────────────────
    # EVENT HANDLERS
    # ─────────────────────────────────────────

    def _format_time(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def _load_video_info(self):
        url = self._url_entry.get().strip()
        if not url:
            messagebox.showwarning("Missing URL", "Please enter a YouTube URL.")
            return

        from app.downloader import is_valid_youtube_url
        if not is_valid_youtube_url(url):
            messagebox.showerror("Invalid URL", "The URL doesn't look like a valid YouTube link.")
            return
            
        self._load_info_btn.configure(state="disabled", text="Loading...")
        
        def fetch():
            try:
                import yt_dlp
                ydl_opts = {'quiet': True, 'no_warnings': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    duration = info.get('duration', 0)
                    
                self.after(0, self._update_sliders, duration)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", f"Failed to get video info: {e}"))
            finally:
                self.after(0, lambda: self._load_info_btn.configure(state="normal", text="Load Info"))
                
        threading.Thread(target=fetch, daemon=True).start()

    def _update_sliders(self, duration: float):
        self._video_duration = duration
        if duration > 0:
            self._start_slider.configure(to=duration)
            self._start_slider.set(0)
            self._end_slider.configure(to=duration)
            self._end_slider.set(duration)
            self._on_start_slider(0)
            self._on_end_slider(duration)
            
    def _on_start_slider(self, value):
        val = float(value)
        end_val = float(self._end_slider.get())
        if val >= end_val:
            val = end_val - 1.0 if end_val >= 1.0 else 0
            self._start_slider.set(val)
        self._start_label.configure(text=f"Start: {self._format_time(val)}")

    def _on_end_slider(self, value):
        val = float(value)
        start_val = float(self._start_slider.get())
        if val <= start_val:
            val = start_val + 1.0 if start_val + 1.0 <= self._video_duration else self._video_duration
            self._end_slider.set(val)
        self._end_label.configure(text=f"End: {self._format_time(val)}")

    def _browse_output(self):
        """Open folder selection dialog."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self._output_dir = Path(folder)
            self._output_label.configure(text=str(self._output_dir), text_color=TEXT)

    def _start_pipeline(self):
        """Validate inputs and start pipeline in background thread."""
        if self._running:
            self._log("⚠️ Pipeline already running!", color="warning")
            return

        url = self._url_entry.get().strip()
        if not url:
            messagebox.showwarning("Missing URL", "Please enter a YouTube URL.")
            return

        from app.downloader import is_valid_youtube_url
        if not is_valid_youtube_url(url):
            messagebox.showerror("Invalid URL", "The URL doesn't look like a valid YouTube link.")
            return

        language = "vi"  # Always Vietnamese — non-negotiable
        
        voice_selection = self._voice_var.get()
        if "Nam" in voice_selection:
            voice_id = "vi-VN-NamMinhNeural"
        else:
            voice_id = "vi-VN-HoaiMyNeural"
            
        trim_start = float(self._start_slider.get())
        trim_end = float(self._end_slider.get())

        self._running = True
        self._run_btn.configure(state="disabled", text="⏳ Processing...")
        self._progress_bar.set(0)
        self._status_label.configure(text="Starting pipeline...", text_color=ACCENT2)
        self._log("═" * 60)
        self._log(f"🎬 Starting pipeline for:\n   {url}", color="accent")
        self._log("═" * 60)

        # Run in background thread to keep UI responsive
        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(url, language, voice_id, trim_start, trim_end),
            daemon=True,
        )
        thread.start()

    def _run_pipeline_thread(self, url: str, language: str, voice_id: str, trim_start: float, trim_end: float):
        """Execute the pipeline in a background thread."""
        try:
            from app.pipeline import run_pipeline, PipelineError

            output_path = run_pipeline(
                youtube_url=url,
                output_dir=self._output_dir,
                language=language,
                voice_id=voice_id,
                trim_start=trim_start,
                trim_end=trim_end,
                progress_callback=self._update_progress,
            )

            self._log_queue.put(("success", f"✅ Video saved: {output_path}"))
            self._log_queue.put(("done", str(output_path)))

        except ValueError as e:
            self._log_queue.put(("error", f"❌ Validation error: {e}"))
            self._log_queue.put(("failed", ""))

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Pipeline exception: {tb}")
            self._log_queue.put(("error", f"❌ Pipeline failed: {e}"))
            self._log_queue.put(("failed", ""))

    def _update_progress(self, percent: float, message: str):
        """Called from pipeline thread to update progress UI."""
        self._log_queue.put(("progress", (percent, message)))

    def _poll_log_queue(self):
        """Process messages from the pipeline thread (runs on main thread)."""
        try:
            while True:
                item = self._log_queue.get_nowait()
                msg_type, data = item

                if msg_type == "progress":
                    pct, msg = data
                    self._progress_bar.set(min(pct, 1.0))
                    self._status_label.configure(text=msg, text_color=ACCENT2)
                    self._log(f"[{pct*100:.0f}%] {msg}")

                elif msg_type == "success":
                    self._log(data, color="success")
                    self._status_label.configure(text="✅ Complete!", text_color=SUCCESS)

                elif msg_type == "error":
                    self._log(data, color="error")
                    self._status_label.configure(text="❌ Failed", text_color=ERROR)

                elif msg_type == "done":
                    self._progress_bar.set(1.0)
                    self._running = False
                    self._run_btn.configure(state="normal", text="▶  Generate TikTok Video")
                    if data:
                        messagebox.showinfo(
                            "Done!",
                            f"Video successfully created!\n\n{data}\n\nReady to post on TikTok / Reels / Shorts! 🎉"
                        )

                elif msg_type == "failed":
                    self._running = False
                    self._run_btn.configure(state="normal", text="▶  Generate TikTok Video")

        except queue.Empty:
            pass

        # Re-schedule
        self.after(100, self._poll_log_queue)

    def _log(self, message: str, color: str = "normal"):
        """Append a message to the log textbox."""
        color_map = {
            "normal": "#C8C8D8",
            "accent": "#FF4757",
            "success": "#2ECC71",
            "error": "#E74C3C",
            "warning": "#F39C12",
        }
        text_color = color_map.get(color, "#C8C8D8")

        self._log_box.configure(state="normal")
        self._log_box.insert("end", message + "\n")
        self._log_box.configure(state="disabled")
        self._log_box.see("end")

    def _clear_logs(self):
        """Clear the log textbox."""
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")
