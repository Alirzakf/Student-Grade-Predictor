#!/bin/bash
# Quick Start Script for Student Grade Predictor (macOS/Linux)

echo ""
echo "============================================"
echo "Student Grade Predictor - Quick Start"
echo "============================================"
echo ""

echo "Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python not found. Please install Python 3.7+"
    exit 1
fi

echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

echo ""
echo "Dependencies installed successfully!"
echo ""
echo "Launching Streamlit dashboard..."
echo "(Dashboard will open at http://localhost:8501)"
echo ""

streamlit run app.py
