@echo off
:: =========================================================
:: CodebookLM v1.0 Launcher
:: Offline AI Codebase Assistant
:: =========================================================
title CodebookLM v1.0

echo Starting CodebookLM v1.0...
echo Understand Any Codebase. Instantly.
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b
)

:: Run Streamlit headlessly and open browser
echo Launching local server...
start "" http://localhost:8501
python -m streamlit run app.py --server.port 8501 --server.headless true

pause
