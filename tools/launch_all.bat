@echo off
REM ===========================================================================
REM  Unattended full-training launcher (multi-day, crash-safe).
REM  Relaunches tools/run_all.py in a loop: if the process dies (crash,
REM  power cut, Windows update...), the next iteration resumes from the
REM  first non-ok stage (inner runners SKIP completed work).
REM  STOP with Ctrl+C.  It is safe at ANY point.
REM
REM  Usage:
REM    launch_all.bat              continue/resume everything
REM    launch_all.bat fresh        archive old results, retrain EVERYTHING
REM    launch_all.bat awgn         lane split process 1 of 2
REM    launch_all.bat rayleigh     lane split process 2 of 2
REM ===========================================================================
setlocal
cd /d "%~dp0\.."

set MODE=%1
if "%MODE%"=="" set MODE=run
if /i "%MODE%"=="fresh" set MODE=run --fresh
if /i "%MODE%"=="awgn" set MODE=run --lane awgn
if /i "%MODE%"=="rayleigh" set MODE=run --lane rayleigh

:loop
echo.
echo ===== %date% %time% :: launching: python tools/run_all.py %MODE% =====
".venv\Scripts\python.exe" tools\run_all.py %MODE%
set RC=%ERRORLEVEL%
if "%RC%"=="0" (
    echo ===== run_all reports COMPLETE - exiting launcher =====
    exit /b 0
)
echo ===== run_all exited with %RC% - relaunching in 60s (Ctrl+C to stop) =====
timeout /t 60 /nobreak >nul
goto loop
