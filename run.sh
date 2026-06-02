#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_python() {
  if [[ -f "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then
    PYTHON_CMD=("$ROOT_DIR/.venv/Scripts/python.exe")
  elif [[ -f "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_CMD=("$ROOT_DIR/.venv/bin/python")
  elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=(python)
  else
    echo "No Python runner found."
    exit 1
  fi
}

resolve_python

if [[ $# -lt 1 ]]; then
  echo "Usage: ./run.sh standard-eval [args...]"
  exit 1
fi

ACTION="$1"
shift

cd "$ROOT_DIR"

case "$ACTION" in
  standard-eval)
    exec "${PYTHON_CMD[@]}" standard_pipeline.py "$@"
    ;;
  *)
    echo "Unknown Cerebro action: $ACTION"
    exit 1
    ;;
esac
