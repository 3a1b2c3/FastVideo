@echo off
:: Run bench_physics_iq_helios.py inside Helios's .venv.
:: Generates Physics-IQ scenarios via Helios (Wan2.1-14B I2V), then you score
:: with score_physics_iq.py inside FastVideo's .venv.
::
:: Usage:
::   run_physics_iq_helios.bat                              full 396-scenario sweep
::   run_physics_iq_helios.bat --limit 4                    smoke
::   run_physics_iq_helios.bat --variant mid                Helios-Mid
::   run_physics_iq_helios.bat --variant distilled          Helios-Distilled (fast)
::   (CPU-offload is the DEFAULT -- needed for 32 GB GPUs)
::   run_physics_iq_helios.bat --high-vram                  keep resident on GPU (>40 GB VRAM)
::   run_physics_iq_helios.bat --videos-dir D:\helios-iq    redirect output
::
:: After generating, score with:
::   "C:\workspace\world\FastVideo\.venv\Scripts\python.exe" ^
::     "C:\workspace\world\FastVideo\score_physics_iq.py" ^
::     --videos-dir "C:\workspace\world\FastVideo\outputs_video\bench_physics_iq_helios"

setlocal enableextensions enabledelayedexpansion

set "HELIOS=C:\workspace\world\Helios"
set "HELIOS_PY=%HELIOS%\.venv\Scripts\python.exe"
set "SCRIPT=%~dp0bench_physics_iq_helios.py"

if not exist "%HELIOS%"      ( echo ERROR: Helios not found at %HELIOS% & exit /b 2 )
if not exist "%HELIOS_PY%"   ( echo ERROR: Helios .venv not built: %HELIOS_PY% & exit /b 2 )
if not exist "%SCRIPT%"      ( echo ERROR: bench script not found at %SCRIPT% & exit /b 2 )

:: Pin CUDA 12.8 -- Helios's torch is cu128.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"

:: Strip ambient venv state -- call Helios's python directly, no uv resolve.
set "VIRTUAL_ENV="
set "PYTHONHOME="
set "PYTHONPATH="
set "UV_PYTHON="
set "UV_PROJECT_ENVIRONMENT="
set "PYTHONIOENCODING=utf-8"

:: Helios's standard env tweaks (mirrored from run_helios.bat).
set "TORCHDYNAMO_DISABLE=1"
set "HF_DEACTIVATE_ASYNC_LOAD=1"
set "HF_HUB_ENABLE_HF_TRANSFER=0"
set "USE_LIBUV=0"
set "TORCH_TCPSTORE_USE_LIBUV=0"

echo ============================================================
echo Physics-IQ on Helios
echo ============================================================
echo   helios env : %HELIOS%
echo   script     : %SCRIPT%
echo   args       : %*
echo ============================================================

:: pushd %TEMP% so no project root is in cwd.
pushd "%TEMP%"
"%HELIOS_PY%" -X utf8 "%SCRIPT%" %*
set EXIT_CODE=%ERRORLEVEL%
popd

if not %EXIT_CODE%==0 (
    echo.
    echo ERROR: bench failed with code %EXIT_CODE%
    exit /b %EXIT_CODE%
)
echo.
echo [done] Score with score_physics_iq.py inside FastVideo's .venv.
exit /b 0
