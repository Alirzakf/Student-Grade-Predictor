@echo off
REM Quick Start Script for Student Grade Predictor (Windows)

echo.
echo ============================================
echo Student Grade Predictor - Quick Start - by Alireza Kafi
echo ============================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.7+
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully!
echo.
echo Launching Streamlit dashboard...
echo (Dashboard will open at http://localhost:8501)
echo.

streamlit run app.py

pause
