#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/Scripts/python.exe" ]]; then
  PYTHON=".venv/Scripts/python.exe"
elif [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
else
  echo "Python was not found. Install Python 3.11 or newer, then try again." >&2
  exit 1
fi

if ! "$PYTHON" -c "import streamlit" >/dev/null 2>&1; then
  echo "Streamlit is not installed in the selected Python environment: $PYTHON" >&2
  echo "Install this project's requirements, then run bash run_app.sh again." >&2
  exit 1
fi

echo "Starting the local app at http://localhost:8501/"
echo "Stop the app with Ctrl+C. This runs locally; publishing is separate."
exec "$PYTHON" -m streamlit run streamlit_app.py
