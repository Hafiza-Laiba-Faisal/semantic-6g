@echo off
REM ============================================================
REM  Wave-A one-command launcher (other PC, Windows)
REM  Usage:  double-click, or:
REM     tools\launch_wave_a.bat              (full run, resumable)
REM     tools\launch_wave_a.bat awgn         (lane: awgn only)
REM     tools\launch_wave_a.bat rayleigh     (lane: rayleigh only)
REM     tools\launch_wave_a.bat status       (see progress only)
REM     tools\launch_wave_a.bat watcher      (SELF-DRIVING mode:
REM                                           train->eval->finalize->
REM                                           dashboard auto-refresh;
REM                                           recommended for 2-day runs)
REM  Safe to re-run anytime: completed runs SKIP automatically.
REM  Internet NOT required after initial setup.
REM ============================================================
setlocal
cd /d "%~dp0.."

echo == [1/6] Governance verify ==
.venv\Scripts\python.exe tools\run_wave_a.py verify
if errorlevel 1 goto :fail

echo == [2/6] Quick smoke (100-step sanity, isolated, auto-tidy) ==
.venv\Scripts\python.exe tools\run_wave_a.py train smoke:fixsnr_10db

set LANE=%1
if "%LANE%"=="watcher" (
    echo == SELF-DRIVING MODE ==
    echo train -^> evaluate -^> finalize -^> dashboard auto-refresh
    .venv\Scripts\python.exe -u tools\wave_a_watcher.py
    goto :end
)
if "%LANE%"=="status" (
    .venv\Scripts\python.exe tools\run_wave_a.py status
    goto :end
)

echo == [3/6] Training 46 runs (resumable, lane='%LANE%') ==
.venv\Scripts\python.exe tools\run_wave_a.py train %LANE%
if errorlevel 1 goto :fail

echo == [4/6] Evaluation (1000-episode diagnostic grid) ==
.venv\Scripts\python.exe tools\run_wave_a.py evaluate %LANE%
if errorlevel 1 goto :fail

echo == [5/6] Finalize summary ==
.venv\Scripts\python.exe tools\run_wave_a.py finalize

echo == [6/6] Dashboard rebuild ==
.venv\Scripts\python.exe tools\build_dashboard.py --open

echo.
echo ALL DONE. dashboard.html updated. Results: results\WAVE_A\
goto :end

:fail
echo.
echo STEP FAILED - fix the error above, then re-run this script.
echo (Completed runs will SKIP; nothing is overwritten.)
:end
pause
