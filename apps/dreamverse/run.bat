@echo off
:: Windows launcher for Dreamverse demo (mirrors scripts/launch/launch_demo.sh).
::
:: Spawns backend + frontend in two separate cmd windows you can Ctrl-C to stop.
::
:: Usage:
::   run.bat                              backend on :8009, frontend (devtools) on :5299
::   run.bat --mock                       use dreamverse-mock-server (no GPU, UI-dev mode)
::   run.bat --no-frontend                backend only
::   run.bat --no-browser                 skip opening the browser
::   run.bat --no-warmup                  skip 3-segment warmup (fast /readyz)
::   run.bat --image <path>               override warmup image (i2v warmup)
::   run.bat --t2v                        force pure t2v warmup (no image)
::
:: Env overrides:
::   BE_PORT=8010 run.bat
::   FRONTEND_MODE=dev run.bat            (no devtools)
::   FRONTEND_MODE=single5s run.bat
::   FASTVIDEO_STARTUP_WARMUP_IMAGE=...   override warmup image (same as --image)
::   CEREBRAS_API_KEY=...                 (LLM features off without these)
::   GROQ_API_KEY=...
::
:: Default resolution: 544x960 @ 121 frames @ 24fps (~5 sec video). Lowered
:: from upstream 1088x1920 to keep warmup under ~2.5 min on a 32 GB 5090
:: bf16. Edit serve_configs/streaming_demo.yaml `default_request.sampling`
:: to change. Per-request client overrides still take effect.

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

:: --- parse flags ---
set MOCK=0
set NO_FRONTEND=0
set NO_BROWSER=0
set NO_WARMUP=0
set "WARMUP_IMAGE_ARG="
:parse
if "%~1"=="" goto args_done
if /I "%~1"=="--mock"        ( set MOCK=1        & shift & goto parse )
if /I "%~1"=="--no-frontend" ( set NO_FRONTEND=1 & shift & goto parse )
if /I "%~1"=="--no-browser"  ( set NO_BROWSER=1  & shift & goto parse )
if /I "%~1"=="--no-warmup"   ( set NO_WARMUP=1   & shift & goto parse )
if /I "%~1"=="--t2v"         ( set "WARMUP_IMAGE_ARG=__EMPTY__" & shift & goto parse )
if /I "%~1"=="--image"       ( set "WARMUP_IMAGE_ARG=%~2" & shift & shift & goto parse )
echo WARN: ignoring unknown arg %~1
shift
goto parse
:args_done

:: --image <path>  -> override the warmup image for this launch.
:: --t2v           -> force pure t2v warmup (clear the image env var).
:: Otherwise FASTVIDEO_STARTUP_WARMUP_IMAGE keeps the default further below.
if defined WARMUP_IMAGE_ARG (
    if "%WARMUP_IMAGE_ARG%"=="__EMPTY__" (
        set "FASTVIDEO_STARTUP_WARMUP_IMAGE="
    ) else (
        set "FASTVIDEO_STARTUP_WARMUP_IMAGE=%WARMUP_IMAGE_ARG%"
    )
)

:: --no-warmup: skip the synthetic 3-segment startup warmup. /readyz flips
:: green within ~2 min of launch (model load only) instead of ~17-25 min.
:: TRADE-OFF: the first real user request takes the JIT/compile hit instead
:: (~5-10 min latency on first generation). After that, subsequent requests
:: are at normal speed. Use this when you want to verify the UI / API loop
:: quickly and don't mind a slow first generation.
if "%NO_WARMUP%"=="1" (
    set "FASTVIDEO_ENABLE_STARTUP_WARMUP=0"
    set "READY_TIMEOUT_S=600"
    echo [run.bat] --no-warmup: FASTVIDEO_ENABLE_STARTUP_WARMUP=0, READY_TIMEOUT_S=600
)

:: --- defaults ---
if not defined BE_PORT set BE_PORT=8009
set FE_PORT=5299
if not defined FRONTEND_MODE set FRONTEND_MODE=devtools

:: NVFP4 quantization needs flashinfer, which has no Windows wheel (the sdist
:: build calls os.symlink, which Windows blocks without Developer Mode, and
:: the underlying CUDA/cutlass code is Linux-focused anyway). Default to
:: skipping NVFP4 so the bf16 fallback loads; set DREAMVERSE_DISABLE_NVFP4=0
:: before launching if you actually have flashinfer working.
if not defined DREAMVERSE_DISABLE_NVFP4 set "DREAMVERSE_DISABLE_NVFP4=1"

set "REPO_ROOT=C:\workspace\world\FastVideo"
set "VENV=%REPO_ROOT%\.venv"
set "WEB=%~dp0web"

:: Warm up the i2v code path (seg=1 with image) instead of pure t2v so the
:: first real user image-to-video request doesn't take the JIT/compile hit.
:: Image is a real-photo aerial beach shot from FastVideo/assets/images/,
:: chosen to match the default warmup prompt ("cinematic drone shot over
:: coastal cliffs at sunrise, golden light, gentle ocean waves").
:: Unset / set empty to fall back to pure t2v warmup.
if not defined FASTVIDEO_STARTUP_WARMUP_IMAGE set "FASTVIDEO_STARTUP_WARMUP_IMAGE=%REPO_ROOT%\assets\images\mixkit-aerial-shot-of-a-beach-with-sea-waves-1087.png"

:: --- pre-flight checks ---
if "%MOCK%"=="1" (
    set "BE_EXE=%VENV%\Scripts\dreamverse-mock-server.exe"
) else (
    set "BE_EXE=%VENV%\Scripts\dreamverse-server.exe"
)
if not exist "!BE_EXE!" (
    echo ERROR: !BE_EXE! not found
    echo Did you install fastvideo[dreamverse] into %VENV%?
    echo Try: uv pip install --python %VENV% -e "%REPO_ROOT%[dreamverse]"
    exit /b 2
)
if "%NO_FRONTEND%"=="0" (
    where npm >nul 2>&1 || ( echo ERROR: npm not on PATH ^(install Node.js^) & exit /b 2 )
    if not exist "%WEB%\node_modules" (
        echo === installing frontend deps ^(one-time^) ===
        pushd "%WEB%"
        call npm ci
        if errorlevel 1 ( echo ERROR: npm ci failed & popd & exit /b 1 )
        popd
    )
)

:: --- env-var warnings ---
if "%CEREBRAS_API_KEY%"=="" echo WARN: CEREBRAS_API_KEY not set ^(LLM-routed features will be disabled^)
if "%GROQ_API_KEY%"==""     echo WARN: GROQ_API_KEY not set ^(LLM-routed features will be disabled^)

:: --- map FRONTEND_MODE → npm script ---
if /I "%FRONTEND_MODE%"=="devtools" (
    set "FE_ENV=set NEXT_PUBLIC_INCLUDE_DEVTOOLS=1"
    set "FE_NPM=dev"
) else if /I "%FRONTEND_MODE%"=="dev" (
    set "FE_ENV=rem no env"
    set "FE_NPM=dev"
) else if /I "%FRONTEND_MODE%"=="single5s" (
    set "FE_ENV=set NEXT_PUBLIC_PRODUCT_MODE=single5s"
    set "FE_NPM=dev"
) else (
    echo ERROR: FRONTEND_MODE must be devtools^|dev^|single5s ^(got %FRONTEND_MODE%^)
    exit /b 2
)

echo ============================================================
echo Dreamverse demo
echo ============================================================
echo   backend     : !BE_EXE! --port %BE_PORT%
echo   backend URL : http://localhost:%BE_PORT%
echo   NVFP4       : DREAMVERSE_DISABLE_NVFP4=%DREAMVERSE_DISABLE_NVFP4%  ^(1=skip ^(bf16 fallback^), 0=use NVFP4^)
if defined FASTVIDEO_STARTUP_WARMUP_IMAGE (
    if not "%FASTVIDEO_STARTUP_WARMUP_IMAGE%"=="" (
        echo   warmup mode : i2v
        echo   warmup img  : %FASTVIDEO_STARTUP_WARMUP_IMAGE%
    ) else (
        echo   warmup mode : t2v ^(no image^)
    )
) else (
    echo   warmup mode : t2v ^(no image^)
)
if "%NO_FRONTEND%"=="0" (
    echo   frontend    : npm run %FE_NPM% ^(mode=%FRONTEND_MODE%^)
    echo   frontend URL: http://localhost:%FE_PORT%
)
echo ============================================================
echo Stop by closing the spawned cmd windows ^(or Ctrl-C in each^).
echo.

:: --- launch backend in its own window ---
start "dreamverse-be :%BE_PORT%" cmd /k ""!BE_EXE!" --port %BE_PORT%"

:: --- wait for backend /readyz to return 200 ---
:: The frontend has a "Dreamverse backend is not reachable. ... wait for /readyz
:: to return 200 before retrying." guard, so opening the browser too early just
:: shows that error. Poll until ready (or 30 min cap) before continuing.
::
:: 1800s gives the full LTX-2 3-segment warmup time to compile its flash-attn
:: + bf16 kernels on Blackwell sm_120. Each segment takes ~5-7 min end-to-end
:: (denoise + VAE + audio decode + save); the synthetic warmup hits seg1 +
:: seg2 + seg1-post-LoRA = ~17-25 min total on a 5090 with current resolution
:: (384x640 / 25 frames). Override via:
::   set READY_TIMEOUT_S=600   then run.bat   (10 min - good if warmup off)
::   set FASTVIDEO_ENABLE_STARTUP_WARMUP=0    skip warmup entirely (first
::                                            user request takes the JIT hit)
if not defined READY_TIMEOUT_S set READY_TIMEOUT_S=1800
echo.
echo Waiting for backend /readyz on :%BE_PORT% ^(up to %READY_TIMEOUT_S%s^) ...
set BE_OK=0
for /L %%i in (1,1,%READY_TIMEOUT_S%) do (
    if !BE_OK!==0 (
        powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:%BE_PORT%/readyz' -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
        if !errorlevel!==0 set BE_OK=1
        if !BE_OK!==1 echo   /readyz 200 after %%i s
        if !BE_OK!==0 timeout /t 1 /nobreak >nul 2>&1
    )
)
if !BE_OK!==0 (
    echo.
    echo ERROR: backend /readyz did not return 200 within %READY_TIMEOUT_S%s.
    echo   Check the "dreamverse-be :%BE_PORT%" window for import / load errors.
    echo   Common causes: missing CEREBRAS_API_KEY/GROQ_API_KEY, broken venv install,
    echo   GPU OOM, or the model snapshot still downloading.
    exit /b 1
)

if "%NO_FRONTEND%"=="1" goto frontend_skipped

:: --- launch frontend in its own window ---
:: cmd /k keeps the window open so logs are visible and you can Ctrl-C.
:: Quoted command runs: set env var, then npm run <script>.
start "dreamverse-fe :%FE_PORT%" cmd /k "cd /d "%WEB%" && %FE_ENV% && npm run %FE_NPM%"

if "%NO_BROWSER%"=="0" (
    :: Give the FE dev server a couple seconds to bind before opening the tab.
    timeout /t 3 /nobreak >nul
    start "" http://localhost:%FE_PORT%
)

:frontend_skipped
echo.
echo --- launched ---
echo   backend  : http://localhost:%BE_PORT%  ^(/healthz, /readyz, /status^)
if "%NO_FRONTEND%"=="0" echo   frontend : http://localhost:%FE_PORT%
echo Stop by closing the spawned cmd windows or Ctrl-C in each.
exit /b 0
