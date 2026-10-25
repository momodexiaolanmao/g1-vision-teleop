@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.ps1 first. See README_zh.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" teleop.py --demo
if errorlevel 1 pause
