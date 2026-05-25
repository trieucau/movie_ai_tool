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

REM ── Add NVIDIA CUDA DLL dirs to PATH (GPU acceleration) ─────
set "NVIDIA_BASE=%~dp0venv\Lib\site-packages\nvidia"
if exist "%NVIDIA_BASE%\cublas\bin"       set "PATH=%NVIDIA_BASE%\cublas\bin;%PATH%"
if exist "%NVIDIA_BASE%\cuda_runtime\bin" set "PATH=%NVIDIA_BASE%\cuda_runtime\bin;%PATH%"
if exist "%NVIDIA_BASE%\cuda_nvrtc\bin"   set "PATH=%NVIDIA_BASE%\cuda_nvrtc\bin;%PATH%"
if exist "%NVIDIA_BASE%\cudnn\bin"        set "PATH=%NVIDIA_BASE%\cudnn\bin;%PATH%"
if exist "%NVIDIA_BASE%\cufft\bin"        set "PATH=%NVIDIA_BASE%\cufft\bin;%PATH%"
if exist "%NVIDIA_BASE%\curand\bin"       set "PATH=%NVIDIA_BASE%\curand\bin;%PATH%"
if exist "%NVIDIA_BASE%\cusolver\bin"     set "PATH=%NVIDIA_BASE%\cusolver\bin;%PATH%"
if exist "%NVIDIA_BASE%\cusparse\bin"     set "PATH=%NVIDIA_BASE%\cusparse\bin;%PATH%"
echo [INFO] CUDA DLL dirs registered for GPU support.

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
