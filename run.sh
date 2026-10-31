#!/bin/bash
# ADDED: Project-scoped runtime and caches; works from any current directory.
set -euo pipefail # ADDED
PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)" # ADDED
cd "$PROJECT_DIR" # ADDED
export HF_HOME="$PROJECT_DIR/.cache/huggingface" # ADDED
export MPLCONFIGDIR="$PROJECT_DIR/.cache/matplotlib" # ADDED
export XDG_CACHE_HOME="$PROJECT_DIR/.cache" # ADDED: Fontconfig cache.
export IPYTHONDIR="$PROJECT_DIR/.cache/ipython" # ADDED
export JUPYTER_CONFIG_DIR="$PROJECT_DIR/.cache/jupyter/config" # ADDED
export JUPYTER_DATA_DIR="$PROJECT_DIR/.venv/share/jupyter" # ADDED
export JUPYTER_RUNTIME_DIR="$PROJECT_DIR/.cache/jupyter/runtime" # ADDED
export MPLBACKEND=Agg # ADDED
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" # ADDED
exec "$PROJECT_DIR/.venv/bin/python" "$@" # ADDED
