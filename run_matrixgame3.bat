@echo off
:: run_matrixgame3.bat -- launch the FastVideo Matrix-Game 3.0 basic example
:: with the Windows-specific env tweaks the upstream example file doesn't set.
:: Mirrors run_matrixgame2_gradio.bat for the matrixgame3 distilled variant.
::
:: Usage:
::   run_matrixgame3.bat                 default run (image from demo URL)
::   run_matrixgame3.bat --no-offline    allow HF Hub network fetch
::                                       (default: HF_HUB_OFFLINE=1)
::
:: Pre-req: HF snapshot in cache. Run once with --no-offline to fetch:
::   run_matrixgame3.bat --no-offline

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

set "EXAMPLE=examples\inference\basic\basic_matrixgame3.py"
if not exist "%~dp0!EXAMPLE!" (
    echo ERROR: example not found: %~dp0!EXAMPLE!
    exit /b 2
)

:: Pre-flight: in offline mode, verify every required component is on disk.
:: Without this the example fails after ~18s with a generic "Unknown model
:: class" trace instead of telling you which file is missing.
:: Skip with HY_NO_MG3_PREFLIGHT=1.
if "%OFFLINE%"=="1" if not defined HY_NO_MG3_PREFLIGHT (
    for /f "delims=" %%S in ('powershell -NoProfile -Command "$r=Get-ChildItem 'C:\Users\kschmid\.cache\huggingface\hub\models--FastVideo--Matrix-Game-3.0-Base-Distilled-Diffusers\snapshots' -Directory -ErrorAction SilentlyContinue ^| Select-Object -First 1 -ExpandProperty FullName; if ($r) { $r } else { 'MISSING' }"') do set "MG3_SNAP=%%S"
    if "!MG3_SNAP!"=="MISSING" (
        echo ERROR: no snapshot dir for FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers in HF cache.
        echo   Run once with --no-offline:  run_matrixgame3.bat --no-offline
        exit /b 2
    )
    set "MG3_OK=1"
    for %%F in (
        "transformer\diffusion_pytorch_model.safetensors"
        "vae\diffusion_pytorch_model.safetensors"
        "text_encoder\model-00001-of-00003.safetensors"
        "light_vae\diffusion_pytorch_model.safetensors"
        "tokenizer\tokenizer.json"
        "model_index.json"
    ) do (
        if not exist "!MG3_SNAP!\%%~F" (
            echo MISSING: !MG3_SNAP!\%%~F
            set "MG3_OK=0"
        )
    )
    if "!MG3_OK!"=="0" (
        echo.
        echo ERROR: Matrix-Game-3.0 incomplete in HF cache ^(see MISSING lines above^).
        echo   Fetch the missing pieces with:  run_matrixgame3.bat --no-offline
        echo   Or to fetch directly:
        echo     .venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers'))"
        echo   Skip this check with: set HY_NO_MG3_PREFLIGHT=1
        exit /b 2
    )
    echo [preflight] all required Matrix-Game-3.0 components present.
)

echo ============================================================
echo FastVideo Matrix-Game 3.0 inference
echo ============================================================
echo   python   : !VENV_PY!
echo   example  : %~dp0!EXAMPLE!
echo   model    : FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers
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
    echo ERROR: matrixgame3 example exited with code !EXIT_CODE!
    if "%OFFLINE%"=="1" (
        echo Hint: model may not be cached. Try once with --no-offline to fetch:
        echo   run_matrixgame3.bat --no-offline
    )
    exit /b !EXIT_CODE!
)

echo.
echo --- done ---
echo videos at: %~dp0video_samples_matrixgame3\
exit /b 0

:help
echo Usage:
echo   run_matrixgame3.bat                 default run (cached model)
echo   run_matrixgame3.bat --no-offline    allow HF Hub network fetch
echo.
echo Env overrides:
echo   GLOO_SOCKET_IFNAME   network adapter name (default: Wi-Fi)
echo.
echo Model: FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers
echo Output: video_samples_matrixgame3\
exit /b 0
