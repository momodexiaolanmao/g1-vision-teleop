@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" teleop.py --robot robots\unitree_g1_twist2\scene.xml --mapping unitree_g1_tracking_mapping.json --auto-calibrate %*
if errorlevel 1 pause
