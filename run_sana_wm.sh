#!/usr/bin/env bash
# Launch the SANA-WM basic example inside WSL/Linux.
#
# Pre-reqs (one-time):
#   1. WSL venv at ~/sana-wm-venv with torch cu128 + FastVideo editable.
#   2. Sana repo cloned at $SANA_REPO_PATH (default C:/workspace/world/Sana
#      reachable via /mnt/c/workspace/world/Sana from WSL).
#   3. SANA-WM_bidirectional snapshot in the HF cache (any cached snapshot OK).
#
# Usage:
#   bash run_sana_wm.sh --image PATH --prompt "..." [--cam_dsl "w-31"] [...]
#
# All flags after this comment forward to examples/inference/basic/basic_sana_wm.py.

set -euo pipefail

VENV="${SANA_WM_VENV:-$HOME/sana-wm-venv}"
SANA_REPO_PATH="${SANA_REPO_PATH:-/mnt/c/workspace/world/Sana}"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ ! -x "$VENV/bin/python" ]; then
    echo "ERROR: venv python not found at $VENV/bin/python" >&2
    echo "  Set SANA_WM_VENV or create one with the Sana deps installed." >&2
    exit 2
fi
if [ ! -d "$SANA_REPO_PATH/diffusion/model" ]; then
    echo "ERROR: Sana repo not found at $SANA_REPO_PATH" >&2
    echo "  Set SANA_REPO_PATH to a clone of NVlabs/Sana." >&2
    exit 2
fi

export SANA_REPO_PATH
export PYTHONPATH="$REPO_ROOT:$SANA_REPO_PATH:${PYTHONPATH:-}"

exec "$VENV/bin/python" "$REPO_ROOT/examples/inference/basic/basic_sana_wm.py" "$@"
