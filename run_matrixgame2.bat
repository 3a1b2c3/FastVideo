@echo off
:: run_matrixgame2.bat -- launch the FastVideo Matrix-Game 2.0 basic example
:: with the Windows-specific env tweaks the upstream example file doesn't set.
:: Mirrors run_matrixgame3.bat for the matrixgame2 base-distilled variant.
::
:: Usage:
::   run_matrixgame2.bat                 default run (image from demo URL)
::   run_matrixgame2.bat --no-offline    allow HF Hub network fetch
::                                       (default: HF_HUB_OFFLINE=1)
::
:: Variant selection: edit MODEL_VARIANT at the top of
::   examples/inference/basic/basic_matrixgame2.py
::   ("base_distilled_model" | "gta_distilled_model" | "templerun_distilled_model")
::
:: Pre-req: HF snapshot in cache. Run once with --no-offline to fetch:
::   run_matrixgame2.bat --no-offline

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "!VENV_PY!" (
    echo ERROR: venv python not found: !VENV_PY!
    exit /b 1
)

:: Strip ambient venv state so the spawned interpreter doesn't graft
:: a different venv's stdlib path onto this one (SRE / _sre mismatch).
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="

:: --- Windows env tweaks ---
:: gloo can't auto-pick a Windows NIC; nudge it at a real adapter.
if not defined GLOO_SOCKET_IFNAME set "GLOO_SOCKET_IFNAME=Wi-Fi"
:: transformers' async shard loader segfaults on Win + sm_120 mid-shard.
set "HF_DEACTIVATE_ASYNC_LOAD=1"
:: hf_transfer's mmap buffers compete with the DiT mmap for Win address space.
set "HF_HUB_ENABLE_HF_TRANSFER=0"
:: PyTorch Win wheels are built without libuv; force gloo's default off.
set "USE_LIBUV=0"
set "TORCH_TCPSTORE_USE_LIBUV=0"
:: UTF-8 stdio so emoji prints don't crash cp1252.
set "PYTHONIOENCODING=utf-8"

:: --- arg parse ---
set "OFFLINE=1"
:parse
if "%~1"=="" goto args_done
if /I "%~1"=="--no-offline" ( set "OFFLINE=0" & shift & goto parse )
if /I "%~1"=="--help"       goto :help
if /I "%~1"=="-h"           goto :help
echo ERROR: unknown arg %~1
exit /b 2
:args_done

:: HF Hub OFFLINE: works around the Py3.12 + Windows thread-join atexit bug
:: in snapshot_download. Pre-req: model already cached. Use --no-offline once
:: to fetch.
if "%OFFLINE%"=="1" (
    set "HF_HUB_OFFLINE=1"
) else (
    set "HF_HUB_OFFLINE="
)

set "EXAMPLE=examples\inference\basic\basic_matrixgame2.py"
if not exist "%~dp0!EXAMPLE!" (
    echo ERROR: example not found: %~dp0!EXAMPLE!
    exit /b 2
)

:: Pre-flight: in offline mode, verify every required component of the default
:: variant (base_distilled) is on disk. Saves ~18s of "Unknown model class"
:: confusion when a file is missing. Skip with HY_NO_MG2_PREFLIGHT=1.
:: Layout: model_index.json + transformer/ + vae/ + image_encoder/ + image_processor/ + scheduler/.
if "%OFFLINE%"=="1" if not defined HY_NO_MG2_PREFLIGHT (
    for /f "delims=" %%S in ('powershell -NoProfile -Command "$r=Get-ChildItem 'C:\Users\kschmid\.cache\huggingface\hub\models--FastVideo--Matrix-Game-2.0-Base-Distilled-Diffusers\snapshots' -Directory -ErrorAction SilentlyContinue ^| Select-Object -First 1 -ExpandProperty FullName; if ($r) { $r } else { 'MISSING' }"') do set "MG2_SNAP=%%S"
    if "!MG2_SNAP!"=="MISSING" (
        echo ERROR: no snapshot dir for FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers in HF cache.
        echo   Run once with --no-offline:  run_matrixgame2.bat --no-offline
        exit /b 2
    )
    set "MG2_OK=1"
    for %%F in (
        "transformer\diffusion_pytorch_model.safetensors"
        "vae\diffusion_pytorch_model.safetensors"
        "image_encoder\model.safetensors"
        "model_index.json"
    ) do (
        if not exist "!MG2_SNAP!\%%~F" (
            echo MISSING: !MG2_SNAP!\%%~F
            set "MG2_OK=0"
        )
    )
    if "!MG2_OK!"=="0" (
        echo.
        echo ERROR: Matrix-Game-2.0-Base-Distilled incomplete in HF cache ^(see MISSING lines above^).
        echo   Fetch the missing pieces with:  run_matrixgame2.bat --no-offline
        echo   Or to fetch directly:
        echo     .venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers'))"
        echo   Skip this check with: set HY_NO_MG2_PREFLIGHT=1
        exit /b 2
    )
    echo [preflight] all required Matrix-Game-2.0-Base-Distilled components present.
)

echo ============================================================
echo FastVideo Matrix-Game 2.0 inference
echo ============================================================
echo   python   : !VENV_PY!
echo   example  : %~dp0!EXAMPLE!
echo   model    : FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers
echo   offline  : %OFFLINE%
echo   gloo if  : !GLOO_SOCKET_IFNAME!
echo ============================================================
echo.

for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "[int][double]::Parse((Get-Date -UFormat %%s))"`) do set "T_START=%%T"
echo Started:  %DATE% %TIME%

"!VENV_PY!" -X utf8 "%~dp0!EXAMPLE!"
set "EXIT_CODE=!ERRORLEVEL!"

for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "[int][double]::Parse((Get-Date -UFormat %%s))"`) do set "T_END=%%T"
echo Finished: %DATE% %TIME%
set /a TOTAL_S=T_END-T_START
echo Elapsed   : %TOTAL_S%s

if not !EXIT_CODE!==0 (
    echo ERROR: matrixgame2 example exited with code !EXIT_CODE!
    if "%OFFLINE%"=="1" (
        echo Hint: model may not be cached. Try once with --no-offline to fetch:
        echo   run_matrixgame2.bat --no-offline
    )
    exit /b !EXIT_CODE!
)

echo.
echo --- done ---
echo videos at: %~dp0video_samples_matrixgame2\
exit /b 0

:help
echo Usage:
echo   run_matrixgame2.bat                 default run (cached model)
echo   run_matrixgame2.bat --no-offline    allow HF Hub network fetch
echo.
echo Env overrides:
echo   GLOO_SOCKET_IFNAME   network adapter name (default: Wi-Fi)
echo.
echo Variants: edit MODEL_VARIANT in examples/inference/basic/basic_matrixgame2.py
echo   base_distilled_model       (default, universal)
echo   gta_distilled_model        (GTA drive)
echo   templerun_distilled_model  (Temple Run)
echo.
echo Model:  FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers
echo Output: video_samples_matrixgame2\
exit /b 0
