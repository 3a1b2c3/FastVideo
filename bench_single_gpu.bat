@echo off
setlocal enableextensions
cd /d "%~dp0"

REM Don't inherit another venv's interpreter state (avoids stdlib/_sre mismatch).
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="

REM Prefer FastVideo's own venv; fall back to whatever python is on PATH.
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo Using interpreter: %PY%
"%PY%" "%~dp0bench_single_gpu.py" %*
exit /b %ERRORLEVEL%
