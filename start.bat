@echo off
chcp 65001 >nul
title AI Studio
cd /d "%~dp0"

echo.
echo   AI Studio
echo   http://127.0.0.1:7860
echo.

python main.py
pause
