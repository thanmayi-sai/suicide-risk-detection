@echo off
title Thanmayi Project - Verify
cd /d "%~dp0"
set "PY="
py -3.11 --version >nul 2>&1 && set "PY=py -3.11"
if not defined PY python --version >nul 2>&1 && set "PY=python"
if not defined PY python3 --version >nul 2>&1 && set "PY=python3"
if not defined PY (
  echo ERROR: Python was not found.
  pause
  exit /b 1
)
%PY% -u tests\run_project_tests.py
echo.
echo Expected final line: passed 37, failed 0, overall_pass true
pause
