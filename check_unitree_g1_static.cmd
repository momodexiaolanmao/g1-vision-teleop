@echo off
cd /d "%~dp0"
call start_unitree_g1.cmd --sample-seconds 20 --sample-output artifacts\unitree_g1_handsfree_static_check.json %*
