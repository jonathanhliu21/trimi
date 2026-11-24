#!/usr/bin/env bash
set -euo pipefail
CONTROL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$CONTROL_ROOT/.venv/bin/python" "$CONTROL_ROOT/scripts/kinesthetic/rollout.py" "$@"
