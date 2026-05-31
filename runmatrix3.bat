@echo off
:: Short alias for run_matrixgame3.bat. Mirrors typing the MIND-side
:: log filename pattern (run_matrix...). Forwards all args verbatim.
cd /d "%~dp0"
call "%~dp0run_matrixgame3.bat" %*
exit /b %ERRORLEVEL%
