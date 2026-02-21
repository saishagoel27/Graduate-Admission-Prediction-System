#!/bin/bash
cd "$(dirname "$0")"

# Check if venv is active
echo "✅ Python used: $(which python)"
echo "✅ Pip used: $(which pip)"

# Check for required packages
pip show streamlit || { echo "❌ streamlit NOT FOUND in venv. Installing..."; pip install streamlit; }
pip show fastapi || { echo "❌ fastapi NOT FOUND. Installing..."; pip install fastapi uvicorn; }

# Check Gemini API Key
if [ -z "$GEMINI_API_KEY" ]; then
  echo "⚠️  GEMINI_API_KEY not set. Attempting to load from .streamlit/secrets.toml..."
  
  # Try to load from secrets.toml
  if [ -f ".streamlit/secrets.toml" ]; then
    export GEMINI_API_KEY=$(grep "GEMINI_API_KEY" .streamlit/secrets.toml | cut -d '"' -f 2)
    echo "✅ Loaded API key from secrets.toml"
  else
    echo "❌ .streamlit/secrets.toml not found."
    echo "Please create it with: GEMINI_API_KEY = \"your-key\""
    exit 1
  fi
fi

# Detect platform
OS="$(uname -s)"
if [[ "$OS" == "Linux" || "$OS" == "Darwin" ]]; then
  PYTHON_PATH="./venv/bin/python"
  STREAMLIT_PATH="./venv/bin/streamlit"
elif [[ "$OS" == "MINGW"* || "$OS" == "CYGWIN"* || "$OS" == "MSYS_NT"* ]]; then
  PYTHON_PATH="./venv/Scripts/python.exe"
  STREAMLIT_PATH="./venv/Scripts/streamlit.exe"
else
  echo "❌ Unsupported OS: $OS"
  exit 1
fi

# Ensure the paths exist
if [ ! -f "$PYTHON_PATH" ]; then
  echo "❌ Python not found at: $PYTHON_PATH"
  exit 1
fi

if [ ! -f "$STREAMLIT_PATH" ]; then
  echo "⚠️  Streamlit not found at: $STREAMLIT_PATH"
  echo "Installing Streamlit..."
  "$PYTHON_PATH" -m pip install streamlit
  
  if [ ! -f "$STREAMLIT_PATH" ]; then
    echo "❌ Failed to install Streamlit"
    exit 1
  fi
fi

# Start FastAPI backend
echo "🚀 Starting FastAPI backend"
"$PYTHON_PATH" -m uvicorn backend.main:app --reload &
BACKEND_PID=$!

# Wait for backend to initialize
sleep 3

# Start Streamlit frontend
echo "🌐 Launching Streamlit frontend"
"$STREAMLIT_PATH" run frontend/app.py

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT