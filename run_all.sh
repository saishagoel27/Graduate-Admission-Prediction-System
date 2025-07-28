#!/bin/bash
cd "$(dirname "$0")"
# run_all.sh - Script to run both FastAPI backend and Streamlit frontend

# Check if venv is active
echo "✅ Python used: $(which python)"
echo "✅ Pip used: $(which pip)"
pip show xgboost || { echo "❌ xgboost NOT FOUND in venv"; exit 1; }

# Check Gemini API Key
if [ -z "$GEMINI_API_KEY" ]; then
  echo "⚠  GEMINI_API_KEY not set. Set it using: export GEMINI_API_KEY=your-key"
  exit 1
fi

# Detect platform
OS="$(uname -s)"
if [[ "$OS" == "Linux" || "$OS" == "Darwin" ]]; then
  PYTHON_PATH="./venv/bin/python"
elif [[ "$OS" == "MINGW"* || "$OS" == "CYGWIN"* || "$OS" == "MSYS_NT"* ]]; then
  PYTHON_PATH="./venv/Scripts/python.exe"
else
  echo "❌ Unsupported OS: $OS"
  exit 1
fi

# Ensure the Python path exists
if [ ! -f "$PYTHON_PATH" ]; then
  echo "❌ Python not found at expected path: $PYTHON_PATH"
  exit 1
fi

# Start FastAPI backend
echo "🚀 Starting FastAPI backend"
"$PYTHON_PATH" -m uvicorn backend.main:app --reload &

# Wait for backend to initialize
sleep 2

# Start Streamlit frontend
echo "🌐 Launching Streamlit frontend"
streamlit run frontend/app.py