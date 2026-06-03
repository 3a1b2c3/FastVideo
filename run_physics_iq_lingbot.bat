@echo off
:: Run bench_physics_iq_lingbot.py inside flashdreams's uv env.
:: Generates Physics-IQ scenarios via lingbot-world-fast, then scores with
:: FastVideo's physics_iq metric.
::
:: Usage:
::   run_physics_iq_lingbot.bat                                   full 396-scenario sweep
::   run_physics_iq_lingbot.bat --limit 4                         smoke
::   run_physics_iq_lingbot.bat --slug lingbot-world-fast-flash   use the flash variant
::   run_physics_iq_lingbot.bat --skip-generation                 re-score existing mp4s
::   run_physics_iq_lingbot.bat --videos-dir D:\fv-iq-outputs     redirect output

setlocal enableextensions enabledelayedexpansion

set "UV_EXE=C:\Users\kschmid\.local\bin\uv.exe"
set "FLASHDREAMS=C:\workspace\world\flashdreams"
set "FLASHDREAMS_PY=%FLASHDREAMS%\.venv\Scripts\python.exe"
set "SCRIPT=%~dp0bench_physics_iq_lingbot.py"

if not exist "%UV_EXE%"         ( echo ERROR: uv not found at %UV_EXE% & exit /b 2 )
if not exist "%FLASHDREAMS%"    ( echo ERROR: flashdreams not found at %FLASHDREAMS% & exit /b 2 )
if not exist "%FLASHDREAMS_PY%" ( echo ERROR: flashdreams .venv not built: %FLASHDREAMS_PY% & exit /b 2 )
if not exist "%SCRIPT%"         ( echo ERROR: bench script not found at %SCRIPT% & exit /b 2 )

:: The bench script is now generation-only -- no fastvideo imports needed.
:: Scoring is decoupled: run score_physics_iq.py inside FastVideo's .venv
:: afterwards. This avoids cross-env dep collisions.

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"

set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="
set "UV_PYTHON="
set "UV_PROJECT_ENVIRONMENT="
set "PYTHONIOENCODING=utf-8"
set "TORCHDYNAMO_DISABLE=1"

echo ============================================================
echo Physics-IQ on lingbot
echo ============================================================
echo   flashdreams env : %FLASHDREAMS%
echo   script          : %SCRIPT%
echo   args            : %*
echo ============================================================

:: Call flashdreams's .venv python DIRECTLY -- avoids `uv run` which would
:: re-resolve the whole workspace and try to build the local fastvideo project
:: (whose pyproject.toml ships a Linux-only `fastvideo-venv_linux\lib64`
:: package dir that breaks setuptools on Windows).
:: Run from %TEMP% so no pyproject.toml in cwd accidentally re-triggers any
:: tooling auto-discovery.
pushd "%TEMP%"
"%FLASHDREAMS_PY%" -X utf8 "%SCRIPT%" %*
set EXIT_CODE=%ERRORLEVEL%
popd

if not %EXIT_CODE%==0 (
    echo.
    echo ERROR: bench failed with code %EXIT_CODE%
    exit /b %EXIT_CODE%
)
echo.
echo [done] check scores.json under the videos-dir for per-scenario + aggregate metrics.
exit /b 0
