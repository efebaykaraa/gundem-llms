#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"
if [[ -n "${DISTILL_PYTHON:-}" ]]; then
  EXPERIMENT_PYTHON="$DISTILL_PYTHON"
elif [[ -x workspace/envs/training/bin/python ]]; then
  EXPERIMENT_PYTHON=workspace/envs/training/bin/python
elif [[ -x data-deepseek/venv-train/bin/python ]]; then
  EXPERIMENT_PYTHON=data-deepseek/venv-train/bin/python
else
  EXPERIMENT_PYTHON=python3
fi
exec "$EXPERIMENT_PYTHON" run_experiments.py "$@"
