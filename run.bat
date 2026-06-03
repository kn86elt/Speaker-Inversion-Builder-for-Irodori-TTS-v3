@echo off
setlocal

set UV=uv
set APP_SCRIPT=webui\app.py
set PORT=7863

set PYTHONUTF8=1
set UV_CACHE_DIR=%~dp0data\uv-cache

where "%UV%" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv was not found on PATH.
    echo [ERROR] Install uv or set PATH so uv.exe can be found.
    pause
    exit /b 1
)

echo [INFO] uv: %UV%

:: Stop only our own app process (find PID of our app.py on the target port)
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":%PORT% "') do (
    for /f "tokens=1" %%b in ('wmic process where "pid=%%a" get commandline 2^>nul ^| findstr /i "webui"') do (
        echo [INFO] Stopping previous instance (PID %%a)
        taskkill /F /PID %%a >nul 2>&1
    )
)
timeout /t 1 /nobreak >nul

echo.
echo [INFO] Starting Speaker Inversion WebUI on port %PORT%
echo [INFO] http://127.0.0.1:%PORT%
echo [INFO] Ctrl+C to stop
echo.

cd /d "%~dp0"

echo [INFO] Syncing minimal WebUI environment...
"%UV%" sync --no-dev
if errorlevel 1 (
    echo [ERROR] uv sync failed.
    pause
    exit /b 1
)

:: Optional: GPU transcription via CUDA (faster-whisper).
:: Requires an NVIDIA GPU with CUDA drivers installed.
:: To enable: uncomment the two lines below.

:: "%UV%" pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
:: set WHISPER_DEVICE=cuda

start /min cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:%PORT%"
"%UV%" run --no-sync python -u "%APP_SCRIPT%" --host 0.0.0.0 --port %PORT%

echo.
pause
