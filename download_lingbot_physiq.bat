@echo off
:: Pre-fetch every artefact needed by bench_physics_iq_lingbot.py:
::   - lingbot-world-fast pipeline (~25 GB total: DiT + VAE + UMT5 + CLIP)
::   - Physics-IQ dataset (~4-6 GB full, ~50 MB with --physics-iq-limit 4)
::
:: Runs inside flashdreams's uv env so lingbot.* imports cleanly.
::
:: Usage:
::   download_lingbot_physiq.bat                            full pull (~30 GB)
::   download_lingbot_physiq.bat --physics-iq-limit 4       smoke (~25 GB lingbot + 50 MB physiq)
::   download_lingbot_physiq.bat --skip-lingbot --physics-iq-limit 4   only physiq smoke
::   download_lingbot_physiq.bat --skip-physics-iq          only lingbot pull
::
:: Env:
::   FASTVIDEO_EVAL_CACHE=<path>   override eval cache root
::   HF_TOKEN=hf_...               if any gated repos
::   FASTVIDEO_PHYSICS_IQ_BUCKET_URL=<url>   internal Physics-IQ mirror

setlocal enableextensions enabledelayedexpansion

set "UV_EXE=C:\Users\kschmid\.local\bin\uv.exe"
set "FLASHDREAMS=C:\workspace\world\flashdreams"
set "SCRIPT=%~dp0download_lingbot_physiq.py"

if not exist "%UV_EXE%" (
    echo ERROR: uv not found at %UV_EXE%
    exit /b 2
)
if not exist "%FLASHDREAMS%\integrations\lingbot" (
    echo ERROR: flashdreams-lingbot plugin not found at %FLASHDREAMS%\integrations\lingbot
    exit /b 2
)
if not exist "%SCRIPT%" (
    echo ERROR: download_lingbot_physiq.py not found at %SCRIPT%
    exit /b 2
)

:: Pin CUDA 12.8 (matches cu128 torch in flashdreams's env).
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"

:: Strip ambient venv state — uv will manage its own.
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="
set "UV_PYTHON="
set "UV_PROJECT_ENVIRONMENT="
set "PYTHONIOENCODING=utf-8"
set "TORCHDYNAMO_DISABLE=1"
set "HF_HUB_ENABLE_HF_TRANSFER=0"

echo ============================================================
echo Pre-fetch: lingbot-world-fast + Physics-IQ
echo ============================================================
echo   flashdreams env : %FLASHDREAMS%
echo   script          : %SCRIPT%
echo   CUDA            : %CUDA_PATH%
echo   args            : %*
echo ============================================================

pushd "%FLASHDREAMS%"
"%UV_EXE%" run --package flashdreams-lingbot python "%SCRIPT%" %*
set EXIT_CODE=%ERRORLEVEL%
popd

if not %EXIT_CODE%==0 (
    echo.
    echo ERROR: pre-fetch failed with code %EXIT_CODE%
    echo Common causes:
    echo   * gated repo: set HF_TOKEN=hf_... then re-run
    echo   * network: check connectivity to huggingface.co and storage.googleapis.com
    echo   * disk: need ~30 GB free for full pull
    exit /b %EXIT_CODE%
)

echo.
echo [done] pre-fetch complete. Next:
echo   run_physics_iq_lingbot.bat --limit 4
exit /b 0
