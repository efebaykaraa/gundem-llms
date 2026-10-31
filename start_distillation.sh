#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "${DISTILL_PYTHON:-python3}" "$SCRIPT_DIR/start_distillation.py" "$@"
