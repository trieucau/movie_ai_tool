# 🎬 Movie AI Tool — YouTube → TikTok Review Generator

Automatically converts any YouTube movie into a **viral TikTok/Reels/Shorts review video** using AI.

> Paste a YouTube URL → get a fully-edited vertical video with AI narration, subtitles, and music.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📥 YouTube Download | High-quality download via `yt-dlp` |
| 🎙️ Transcription | Fast AI transcription with `faster-whisper` |
| 🤖 AI Script | GPT-powered viral TikTok script generation |
| ✂️ Auto Editing | Keyword-matched clip selection + cutting |
| 📱 Vertical Format | Smart 9:16 center-crop conversion |
| 🗣️ AI Voiceover | Free Edge TTS or ElevenLabs |
| 📝 Subtitles | Karaoke-style word-by-word burn-in |
| 🎵 Background Music | Auto-mixed cinematic music |
| 🖥️ GUI | Dark modern UI with progress tracking |

---

## 🖥️ Requirements

- **Windows 10/11**
- **Python 3.11+** → [Download](https://www.python.org/downloads/)
- **ffmpeg** → see install below
- **OpenAI API key** → [Get here](https://platform.openai.com/api-keys)

---

## ⚡ Quick Start

### 1. Install Python 3.11+

Download from [python.org](https://www.python.org/downloads/) and check **"Add to PATH"** during install.

```
python --version   # should print 3.11+
```

### 2. Install ffmpeg

**Option A — winget (recommended):**
```bat
winget install Gyan.FFmpeg
```

**Option B — manual:**
1. Download from https://ffmpeg.org/download.html (Windows build)
2. Extract to `C:\ffmpeg\`
3. Add `C:\ffmpeg\bin` to your system PATH

Verify: `ffmpeg -version`

### 3. Clone or Download the Tool

```bat
git clone https://github.com/yourrepo/movie-ai-tool.git
cd movie-ai-tool
```

Or extract the ZIP and open a terminal in the folder.

### 4. Configure API Keys

Copy `.env` and fill in your keys:

```bat
notepad .env
```

Minimum required:
```
OPENAI_API_KEY=sk-...your key here...
```

### 5. Run the Tool

**Double-click `run.bat`** — it will:
- Create a virtual environment automatically
- Install all dependencies
- Launch the GUI

Or manually:
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## 🎮 How to Use

1. **Launch** `run.bat`
2. **Paste** a YouTube movie URL
3. **Select** output folder (optional)
4. **Choose** language: English or Vietnamese
5. **Click** "▶ Generate TikTok Video"
6. Wait ~5-15 minutes depending on movie length
7. 🎉 Find your video in the `output/` folder!

---

## ⚙️ Configuration

All settings are in `.env`:

```env
# AI Models
OPENAI_API_KEY=sk-...         # Required
OPENAI_MODEL=gpt-4o-mini      # Or gpt-4o for better quality

# Voice
TTS_ENGINE=edge               # edge (free) or elevenlabs (premium)
EDGE_VOICE=en-US-GuyNeural    # Voice name for Edge TTS

# Transcription
WHISPER_MODEL=base            # tiny/base/small/medium/large-v3
WHISPER_DEVICE=cpu            # cpu or cuda (if NVIDIA GPU)

# Output
MAX_VIDEO_DURATION=90         # Max output video length in seconds
```

### Available Edge TTS Voices

| Language | Voice Name |
|---|---|
| English (US Male) | `en-US-GuyNeural` |
| English (US Female) | `en-US-JennyNeural` |
| English (UK Male) | `en-GB-RyanNeural` |
| Vietnamese (Male) | `vi-VN-NamMinhNeural` |
| Vietnamese (Female) | `vi-VN-HoaiMyNeural` |

### GPU Acceleration (NVIDIA)

If you have an NVIDIA GPU with CUDA:
```env
WHISPER_DEVICE=cuda
WHISPER_COMPUTE=float16
USE_GPU=true
```

---

## 🎵 Adding Background Music

Drop `.mp3` or `.wav` files into `assets/music/`.  
The tool will randomly pick one and auto-mix it at low volume.

---

## 📁 Project Structure

```
movie_ai_tool/
├── app/
│   ├── downloader/      # YouTube download (yt-dlp)
│   ├── transcription/   # Audio → text (faster-whisper)
│   ├── llm/             # AI script generation (OpenAI)
│   ├── clipper/         # Scene matching + video cutting
│   ├── subtitle/        # Karaoke subtitle generation
│   ├── voice/           # TTS generation
│   ├── render/          # Final video compositing
│   ├── utils/           # Logging, file helpers
│   ├── config/          # Settings management
│   ├── pipeline.py      # Main orchestrator
│   └── gui.py           # GUI application
├── assets/
│   ├── music/           # Background music files (add your own)
│   ├── fonts/           # Custom fonts (optional)
│   └── overlays/        # Overlay images (optional)
├── input/               # Manual input files
├── output/              # Final videos saved here
├── temp/                # Intermediate processing files
├── logs/                # Log files
├── main.py              # Entry point
├── requirements.txt
├── .env                 # API keys & config
└── run.bat              # Double-click launcher
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `ffmpeg not found` | Install ffmpeg and add to PATH (see Step 2) |
| `OPENAI_API_KEY not set` | Edit `.env` and add your key |
| `faster-whisper` slow | Set `WHISPER_MODEL=tiny` for speed |
| Video download fails | Update yt-dlp: `pip install -U yt-dlp` |
| No audio in output | Ensure voice.mp3 was generated (check logs/) |
| Subtitle not showing | Ensure Arial font is installed on Windows |

---

## 📄 License

MIT License. Free for personal and commercial use.

---

## 🙏 Credits

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube downloader
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Fast Whisper transcription
- [edge-tts](https://github.com/rany2/edge-tts) — Free Microsoft Edge TTS
- [OpenAI](https://openai.com) — GPT script generation
- [customtkinter](https://github.com/TomSchimansky/CustomTkinter) — Modern GUI
- [ffmpeg](https://ffmpeg.org) — Video processing
