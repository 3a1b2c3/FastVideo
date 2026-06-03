@echo off
:: Pre-fetch the Physics-IQ benchmark dataset into the FastVideo eval cache.
::
:: Source: DeepMind public GCS bucket https://storage.googleapis.com/physics-iq-benchmark
:: Destination: %FASTVIDEO_EVAL_CACHE%\datasets\physics_iq\  (default: %USERPROFILE%\.cache\fastvideo\eval\)
::
:: Why pre-fetch: bench_physics_iq.py auto-downloads on first use, but pulling
:: the whole corpus up front is friendlier on long eval runs (no mid-eval stalls,
:: easier to resume).
::
:: Estimated size:
::   --limit 4   : ~50 MB         (smoke run)
::   --limit 20  : ~250 MB        (debug subset, varied categories)
::   --limit 60  : ~750 MB        (one perspective view across all 66 setups)
::   full        : ~4-6 GB        (all 396 scenarios x take1+take2 + masks + switch frames)
::
:: Usage:
::   download_physics_iq.bat                              full pull (~4-6 GB)
::   download_physics_iq.bat --limit 4                    smoke ~50 MB
::   download_physics_iq.bat --limit 60                   one-view sweep
::   download_physics_iq.bat --dry-run                    print plan, don't fetch
::   download_physics_iq.bat --cache-dir D:\fv-eval       redirect away from C:\
::
:: Optional env overrides:
::   FASTVIDEO_EVAL_CACHE=<path>             cache root (else ~/.cache/fastvideo/eval)
::   FASTVIDEO_PHYSICS_IQ_BUCKET_URL=<url>   internal mirror of the DeepMind bucket
::   FASTVIDEO_PY=<python.exe>               specific Python with fastvideo installed

setlocal enableextensions enabledelayedexpansion

set "LIMIT="
set "DRY_RUN=0"
set "CACHE_DIR="

:parse
if "%~1"=="" goto args_done
if /I "%~1"=="--limit"      ( set "LIMIT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--dry-run"    ( set "DRY_RUN=1" & shift & goto parse )
if /I "%~1"=="--cache-dir"  ( set "CACHE_DIR=%~2" & shift & shift & goto parse )
if /I "%~1"=="--help"       goto :help
if /I "%~1"=="-h"           goto :help
echo ERROR: unknown arg %~1
exit /b 2
:args_done

:: Pick a Python that has fastvideo installed.
:: Priority: FASTVIDEO_PY -> FastVideo's own .venv -> Helios venv (fallback, may not have fastvideo).
if defined FASTVIDEO_PY (
    set "PY=%FASTVIDEO_PY%"
) else if exist "C:\workspace\world\FastVideo\.venv\Scripts\python.exe" (
    set "PY=C:\workspace\world\FastVideo\.venv\Scripts\python.exe"
) else if exist "C:\workspace\world\Helios\.venv\Scripts\python.exe" (
    set "PY=C:\workspace\world\Helios\.venv\Scripts\python.exe"
) else (
    echo ERROR: no Python with fastvideo found.
    echo Set FASTVIDEO_PY=^<path-to-python.exe^> for a venv with fastvideo installed.
    exit /b 2
)

if defined CACHE_DIR set "FASTVIDEO_EVAL_CACHE=%CACHE_DIR%"

echo ============================================================
echo Physics-IQ pre-fetch
echo ============================================================
echo   py                : %PY%
echo   limit             : !LIMIT!   ^(empty = all 396^)
echo   FASTVIDEO_EVAL_CACHE : !FASTVIDEO_EVAL_CACHE!   ^(empty = ~/.cache/fastvideo/eval^)
echo   FASTVIDEO_PHYSICS_IQ_BUCKET_URL : !FASTVIDEO_PHYSICS_IQ_BUCKET_URL!
echo   dry-run           : %DRY_RUN%
echo ============================================================

if "%DRY_RUN%"=="1" (
    echo [dry-run] would call get_dataset^("physics_iq", limit=!LIMIT!^)
    echo [dry-run] which auto-fetches missing take-1, take-2, masks, switch-frames
    exit /b 0
)

:: Pin CUDA 12.8 to the front of PATH so torch_cuda.dll loads cudart64_12.dll
:: from v12.8 rather than v13.0. Same recipe MIND drive_*.bat uses.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"

:: Strip ambient venv pollution before spawning the chosen Python.
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="
set "UV_PYTHON="
set "UV_PROJECT_ENVIRONMENT="
set "PYTHONIOENCODING=utf-8"
set "TORCHDYNAMO_DISABLE=1"

:: pushd to TEMP so no project root is in cwd (avoids any uv/poetry pickup).
pushd "%TEMP%"

if defined LIMIT (
    "%PY%" -X utf8 -c "import sys; from fastvideo.eval.datasets import get_dataset; ds = get_dataset('physics_iq', limit=int(sys.argv[1])); rows = list(ds); print(f'fetched {len(rows)} scenarios into {ds.dataset_dir}')" "!LIMIT!"
) else (
    "%PY%" -X utf8 -c "from fastvideo.eval.datasets import get_dataset; ds = get_dataset('physics_iq'); rows = list(ds); print(f'fetched {len(rows)} scenarios into {ds.dataset_dir}')"
)
set EXIT_CODE=%ERRORLEVEL%
popd

if not %EXIT_CODE%==0 (
    echo.
    echo ERROR: pre-fetch failed with code %EXIT_CODE%
    echo Common causes:
    echo   * fastvideo not installed in %PY%
    echo     fix: cd FastVideo ^&^& uv pip install -e .[eval^]
    echo   * network: DeepMind GCS bucket unreachable from this host
    echo   * disk: ensure target drive has 5+ GB free for full pull
    exit /b %EXIT_CODE%
)

echo.
echo ============================================================
echo Inventory:
echo ============================================================
powershell -NoProfile -Command "$root = if ($env:FASTVIDEO_EVAL_CACHE) { $env:FASTVIDEO_EVAL_CACHE } else { Join-Path $env:USERPROFILE '.cache\fastvideo\eval' }; $d = Join-Path $root 'datasets\physics_iq'; if (Test-Path $d) { $sum = (Get-ChildItem $d -Recurse -File | Measure-Object -Property Length -Sum); Write-Host ('  total: {0:N1} MB across {1} files' -f ($sum.Sum/1MB), $sum.Count); Write-Host ('  dir  : {0}' -f $d) } else { Write-Host ('  (not created at ' + $d + ')') }"

echo.
echo Next: run the bench script
echo   "%PY%" examples\inference\eval\bench_physics_iq.py --limit 4 --num-gpus 1
exit /b 0

:help
echo Usage:
echo   download_physics_iq.bat [--limit N] [--cache-dir PATH] [--dry-run]
echo.
echo Env overrides:
echo   FASTVIDEO_PY=^<python.exe^>            Force a specific Python ^(must have fastvideo installed^)
echo   FASTVIDEO_EVAL_CACHE=^<path^>          Override the eval cache root
echo   FASTVIDEO_PHYSICS_IQ_BUCKET_URL=^<url^>  Internal mirror of DeepMind bucket
echo.
echo Examples:
echo   download_physics_iq.bat                          full pull, all 396 scenarios ^(~4-6 GB^)
echo   download_physics_iq.bat --limit 4                smoke run, ~50 MB
echo   download_physics_iq.bat --cache-dir D:\fv-eval   redirect away from C:\
exit /b 0
