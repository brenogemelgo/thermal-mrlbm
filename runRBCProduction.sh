#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# All options are forwarded, including --hdf5-root /path/to/separate/storage.
exec python3 -u "$root/scripts/rbc_production.py" "$@"
