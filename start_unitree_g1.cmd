@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" teleop.py --robot robots\unitree_g1\scene_physics.xml --mapping unitree_g1_walking_mapping.json --auto-calibrate %*
if errorlevel 1 pause
