@echo off
rem =====================================================================
rem  semantic-6g - pinned environment activator (Windows)
rem
rem  The venv intentionally lives OUTSIDE the repo:
rem      C:\Users\LENOVO\venvs\sem6g
rem  because this repo path is very long and torch ships ~8.5k header
rem  files under Lib\site-packages\torch\include\...  Windows' 260-char
rem  MAX_PATH limit would silently truncate the install.
rem
rem  Usage (in cmd.exe, inside this repo):
rem      activate.bat
rem      python main.py smoke M1 task awgn 1 42
rem
rem  Usage (in PowerShell):
rem      cmd /k "%CD%\activate.bat"      (or open a new cmd window)
rem
rem  Environment: Python 3.9.13 + torch 2.4.1+cpu + numpy 1.26.4
rem               + matplotlib 3.9.4  (frozen thesis stack, CPU-only)
rem =====================================================================
call "C:\Users\LENOVO\venvs\sem6g\Scripts\activate.bat" || (
    echo [ERROR] venv not found at C:\Users\LENOVO\venvs\sem6g
    echo         Recreate it with:  py -3.9 -m venv C:\Users\LENOVO\venvs\sem6g
    exit /b 1
)
echo [OK] semantic-6g environment active: %VIRTUAL_ENV%
python -c "import sys, torch, numpy, matplotlib; print('python %s | torch %s | numpy %s | matplotlib %s' % (sys.version.split()[0], torch.__version__, numpy.__version__, matplotlib.__version__))"
