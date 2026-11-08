#!/usr/bin/env bash
# Explicit execution entry point. This file is never sourced by the solver.
set -euo pipefail
validation_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${validation_root}/scripts/rbc_validation_campaign.py" "$@"
