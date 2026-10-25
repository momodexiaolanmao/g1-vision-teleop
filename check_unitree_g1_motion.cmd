@echo off
cd /d "%~dp0"
call start_unitree_g1.cmd --sample-seconds 25 --sample-kind motion --sample-output artifacts\unitree_g1_motion_check.json %*
