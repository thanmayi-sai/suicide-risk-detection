@echo off
title Thanmayi Project - Start Application
cd /d "%~dp0"
set "PY="
py -3.11 --version >nul 2>&1 && set "PY=py -3.11"
if not defined PY python --version >nul 2>&1 && set "PY=python"
if not defined PY python3 --version >nul 2>&1 && set "PY=python3"
if not defined PY (
  echo ERROR: Python was not found. Run SETUP_PROJECT.bat after installing Python.
  pause
  exit /b 1
)
echo Starting project...
echo Browser will open automatically at http://localhost:8501
%PY% -m streamlit run app.py
pause
