@echo off
setlocal

set VENV_PYTHON=C:\usr\sd\Irodori-TTS-v3\.venv\Scripts\python.exe

if exist "%VENV_PYTHON%" (
    set PYTHON=%VENV_PYTHON%
) else (
    echo [WARN] venv not found: %VENV_PYTHON%
    echo [WARN] Using system python
    set PYTHON=python
)

set PYTHONUTF8=1

echo [INFO] Python: %PYTHON%
echo.

cd /d "%~dp0"
"%PYTHON%" -u webui\app.py --host 0.0.0.0 --port 7863

echo.
pause
