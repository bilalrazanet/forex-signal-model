@echo off
title ForexAI Signal Dashboard Launcher
echo ===================================================
echo   Starting ForexAI Signal Dashboard...
echo   (This will run in the background)
echo ===================================================
cd /d "d:\Forex trading\forex_signal_model"
echo Activating virtual environment...
call .venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b %errorlevel%
)
echo Starting Flask Server...
python app.py
if %errorlevel% neq 0 (
    echo [ERROR] Flask server crashed!
    pause
)
