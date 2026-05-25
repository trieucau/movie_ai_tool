@echo off
title Movie AI Tool - TikTok Review Generator
color 0A

echo.
echo  =========================================
echo   Movie AI Tool - TikTok Review Generator
echo  =========================================
echo.

REM ── Check if venv exists ──────────────────────────────────────
if not exist "venv\Scripts\activate.bat" (
    echo [SETUP] Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        echo         Make sure Python 3.11+ is installed and on PATH.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM ── Activate venv ────────────────────────────────────────────
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM ── Install requirements if needed ───────────────────────────
python -c "import customtkinter" 2>nul
if errorlevel 1 (
    echo [SETUP] Installing dependencies - first run may take a few minutes...
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed.
)

REM ── Check ffmpeg ─────────────────────────────────────────────
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo.
    echo [WARNING] ffmpeg not found on PATH!
    echo          Please install ffmpeg and add it to your PATH.
    echo          Download: https://ffmpeg.org/download.html
    echo          Or install via: winget install ffmpeg
    echo.
    echo  The tool requires ffmpeg to process video.
    echo  Press any key to continue anyway - may fail without ffmpeg...
    pause >nul
)

REM ── Check .env ───────────────────────────────────────────────
if not exist ".env" (
    echo [WARNING] .env file not found!
    echo          Please copy .env.example to .env and set your API keys.
    pause
    exit /b 1
)

REM ── Launch app ───────────────────────────────────────────────
echo.
echo [INFO] Launching Movie AI Tool...
echo.
python main.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application crashed. Check logs/ folder for details.
    pause
)

deactivate
