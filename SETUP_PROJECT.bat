@echo off
title Thanmayi Project - First Time Setup
cd /d "%~dp0"
set "PY="
py -3.11 --version >nul 2>&1 && set "PY=py -3.11"
if not defined PY python --version >nul 2>&1 && set "PY=python"
if not defined PY python3 --version >nul 2>&1 && set "PY=python3"
if not defined PY (
  echo.
  echo ERROR: Python was not found.
  echo Install Python 3.11 from https://www.python.org/downloads/
  echo During installation enable "Add Python to PATH".
  pause
  exit /b 1
)
echo Using: %PY%
%PY% -m pip install --upgrade pip
if errorlevel 1 goto :fail
%PY% -m pip install -r requirements.txt
if errorlevel 1 goto :fail
echo.
echo SETUP COMPLETED SUCCESSFULLY.
echo Next: double-click START_PROJECT.bat
pause
exit /b 0
:fail
echo.
echo SETUP FAILED. Read PROJECT_RUN_GUIDE_BEGINNER.pdf - Troubleshooting section.
pause
exit /b 1
