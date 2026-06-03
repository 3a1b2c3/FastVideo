@echo off
:: Smoke test the Physics-IQ dataset machinery in FastVideo.
::
:: Validates fastvideo install, dataset auto-fetch, and that the first
:: row's referenced files actually exist on disk. Pulls a tiny subset
:: (default 2 scenarios, ~25 MB) so it's safe to run ad-hoc.
::
:: Usage:
::   smoke_physics_iq.bat                       limit=2 (~25 MB)
::   smoke_physics_iq.bat --limit 4             limit=4 (~50 MB)
::   smoke_physics_iq.bat --cache-dir D:\fv-eval   redirect cache off C:
::   smoke_physics_iq.bat --no-fetch            assume assets already on disk
::
:: Env overrides:
::   FASTVIDEO_PY=<python.exe>   Force a specific Python (must have fastvideo installed)

setlocal enableextensions enabledelayedexpansion

cd /d "%~dp0"

:: Pick a Python that has fastvideo installed.
:: Priority: FASTVIDEO_PY -> FastVideo's own .venv -> Helios venv (likely no fastvideo, errors clean).
if defined FASTVIDEO_PY (
    set "PY=%FASTVIDEO_PY%"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
) else if exist "C:\workspace\world\Helios\.venv\Scripts\python.exe" (
    set "PY=C:\workspace\world\Helios\.venv\Scripts\python.exe"
) else (
    echo ERROR: no Python found.
    echo Set FASTVIDEO_PY=^<path-to-python.exe^> for a venv with fastvideo installed.
    exit /b 2
)

:: Pin CUDA 12.8 to the front of PATH so torch_cuda.dll loads cudart64_12.dll
:: from v12.8 rather than v13.0 (which loses to cu128-built torch's ABI).
:: This is the same recipe drive_*.bat uses for the MIND drivers.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"

:: Strip ambient venv pollution.
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="
set "UV_PYTHON="
set "UV_PROJECT_ENVIRONMENT="
set "PYTHONIOENCODING=utf-8"
set "TORCHDYNAMO_DISABLE=1"

echo [bat]    python  = %PY%
echo [bat]    CUDA    = %CUDA_PATH%

"%PY%" -X utf8 "%~dp0smoke_physics_iq.py" %*
exit /b %ERRORLEVEL%
