@echo off
:: Download FastVideo SSIM reference videos for local regression testing.
::
:: Source: HF dataset FastVideo/ssim-reference-videos
:: Destination: fastvideo\tests\ssim\reference_videos\
::
:: Estimated size:
::   default tier, L40S only:             ~150 MB
::   default tier, all GPU variants:      ~300 MB
::   full_quality tier, all GPU variants: ~1 GB
::   everything (full CI parity):         ~1.2 - 1.5 GB
::
:: Usage:
::   download_ssim_references.bat                            full pull (~1.2 GB)
::   download_ssim_references.bat --tier default             skip full_quality (~300 MB)
::   download_ssim_references.bat --gpu L40S                 single GPU references only
::   download_ssim_references.bat --gpu L40S --tier default  smallest pull (~150 MB)
::   download_ssim_references.bat --dry-run                  print plan, don't download
::
:: Notes:
::   - Uses an existing Python that ships huggingface_hub instead of `uv run`.
::     uv run inside the FastVideo repo triggers a project sync that pulls ROCm
::     composable_kernel sources and explodes on Windows MAX_PATH=260. Skipping
::     uv sidesteps that entire mess.
::   - On RTX 5090 (Blackwell), test_*_similarity.py falls back to L40S
::     references, so at minimum:
::         download_ssim_references.bat --gpu L40S --tier default

setlocal enableextensions enabledelayedexpansion

set "TIER=all"
set "GPU="
set "DRY_RUN=0"
set "REPO=FastVideo/ssim-reference-videos"

:parse
if "%~1"=="" goto args_done
if /I "%~1"=="--tier"     ( set "TIER=%~2" & shift & shift & goto parse )
if /I "%~1"=="--gpu"      ( set "GPU=%~2" & shift & shift & goto parse )
if /I "%~1"=="--repo"     ( set "REPO=%~2" & shift & shift & goto parse )
if /I "%~1"=="--dry-run"  ( set "DRY_RUN=1" & shift & goto parse )
if /I "%~1"=="--help"     goto :help
if /I "%~1"=="-h"         goto :help
echo ERROR: unknown arg %~1
exit /b 2
:args_done

:: Pick a Python that already ships huggingface_hub. Priority:
::   1. FASTVIDEO_PY env var (caller-supplied)
::   2. Helios venv (cp311 + cu128 + hf_hub; the most common setup on this box)
::   3. MIND venv
::   4. RealWonder venv (rarely has hf_hub but worth trying)
if defined FASTVIDEO_PY (
    set "PY=%FASTVIDEO_PY%"
) else if exist "C:\workspace\world\Helios\.venv\Scripts\python.exe" (
    set "PY=C:\workspace\world\Helios\.venv\Scripts\python.exe"
) else if exist "C:\workspace\world\MIND\.venv\Scripts\python.exe" (
    set "PY=C:\workspace\world\MIND\.venv\Scripts\python.exe"
) else if exist "C:\workspace\world\RealWonder\.venv\Scripts\python.exe" (
    set "PY=C:\workspace\world\RealWonder\.venv\Scripts\python.exe"
) else (
    echo ERROR: no Python with huggingface_hub found.
    echo Set FASTVIDEO_PY=^<path-to-python.exe^> for a venv that has huggingface_hub installed.
    exit /b 2
)

set "DEST=%~dp0fastvideo\tests\ssim"
if not exist "%DEST%" (
    echo ERROR: ssim test dir not found: %DEST%
    echo Run this bat from the FastVideo repo root, or move the file there.
    exit /b 2
)

:: Build the allow_patterns glob for snapshot_download.
:: HF dataset layout (per fastvideo/tests/ssim/reference_videos_cli.py):
::   reference_videos/<tier>/<GPU>_reference_videos/<model_id>/<backend>/<file>
if /I "%TIER%"=="all" (
    if defined GPU (
        set "PATTERN=reference_videos/*/%GPU%_reference_videos/**"
    ) else (
        set "PATTERN=reference_videos/**"
    )
) else (
    if defined GPU (
        set "PATTERN=reference_videos/%TIER%/%GPU%_reference_videos/**"
    ) else (
        set "PATTERN=reference_videos/%TIER%/**"
    )
)

echo ============================================================
echo SSIM reference videos download
echo ============================================================
echo   repo    : %REPO%
echo   tier    : %TIER%
echo   gpu     : !GPU!  ^(empty = all^)
echo   dest    : %DEST%
echo   pattern : !PATTERN!
echo   py      : %PY%
echo   dry-run : %DRY_RUN%
echo ============================================================

if "%DRY_RUN%"=="1" (
    echo [dry-run] would snapshot_download %REPO% -^> %DEST%\reference_videos\
    echo [dry-run] allow_patterns = !PATTERN!
    exit /b 0
)

:: Strip ambient venv state so the chosen Python loads its own stdlib cleanly.
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="
set "UV_PYTHON="
set "UV_PROJECT_ENVIRONMENT="
set "PYTHONIOENCODING=utf-8"

:: Pass args via argv to keep quoting simple. Run from a NON-project dir so
:: no uv / poetry / pdm picks up FastVideo's pyproject.toml and tries to sync
:: composable_kernel etc.
pushd "%TEMP%"

"%PY%" -X utf8 -c "import sys; from huggingface_hub import snapshot_download; repo, dest, pat = sys.argv[1:4]; p = snapshot_download(repo_id=repo, repo_type='dataset', local_dir=dest, allow_patterns=[pat]); print('downloaded to:', p)" "%REPO%" "%DEST%" "!PATTERN!"

set EXIT_CODE=%ERRORLEVEL%
popd

if not %EXIT_CODE%==0 (
    echo.
    echo ERROR: download failed with code %EXIT_CODE%
    echo Common causes:
    echo   * gated dataset      : set HF_TOKEN=hf_... then re-run
    echo   * huggingface_hub missing in %PY% : set FASTVIDEO_PY to a venv with it
    echo   * network            : retry, or pre-set HF_HUB_OFFLINE=0
    exit /b %EXIT_CODE%
)

echo.
echo ============================================================
echo Inventory ^(reference_videos\ under %DEST%^):
echo ============================================================
powershell -NoProfile -Command "$base='%DEST%\reference_videos'; if (Test-Path $base) { $sum = (Get-ChildItem $base -Recurse -File | Measure-Object -Property Length -Sum); Write-Host ('  total: {0:N1} MB across {1} files' -f ($sum.Sum/1MB), $sum.Count); Get-ChildItem $base -Directory | ForEach-Object { $n=(Get-ChildItem $_.FullName -Recurse -File | Measure-Object -Property Length -Sum).Sum; Write-Host ('  {0,-30} {1,8:N1} MB' -f $_.Name, ($n/1MB)) } } else { Write-Host '  (reference_videos dir not created)' }"

echo.
echo To smoke-test one model against these references:
echo   "%PY%" -m pytest "%~dp0fastvideo\tests\ssim\test_matrixgame3_similarity.py" -v
echo   ^(note: this requires fastvideo installed in that venv too -- typically run from a FastVideo dev env^)
echo.
exit /b 0

:help
echo Usage:
echo   download_ssim_references.bat [--tier {default^|full_quality^|all}] [--gpu {A40^|L40S^|H100^|H200}] [--dry-run]
echo.
echo Env overrides:
echo   FASTVIDEO_PY=^<path-to-python.exe^>  Force a specific Python ^(must have huggingface_hub^)
echo   HF_TOKEN=hf_...                     If the dataset is gated
echo   FASTVIDEO_SSIM_REFERENCE_HF_REPO=^<repo^>  Override the HF dataset
echo.
echo Examples:
echo   download_ssim_references.bat                              full pull, all tiers + GPUs ^(~1.2 GB^)
echo   download_ssim_references.bat --tier default               skip full_quality ^(~300 MB^)
echo   download_ssim_references.bat --gpu L40S --tier default    smallest, Blackwell-compatible ^(~150 MB^)
exit /b 0
