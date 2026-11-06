#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "$0")/../synthetic_mfx_stage3_20261104/runtime.sh" "$@"
