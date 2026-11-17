#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: ./runConvertVTI.sh <path/to/hdf5> [convertVTI options]

Builds the RBC solver and convertVTI using make all, then converts HDF5 volumes.
The solver is built with --build-only; no simulation is started.
Relative input and results paths use your current working directory.

Converter options (forwarded unchanged):
  --jobs N              Parallel conversion workers (default: up to 4)
  --results-root DIR    Results root (default: results)
  --step N              Convert only this timestep; may be repeated
  --encoding base64|raw XML-safe Base64 by default
  --skip-existing       Resume after checking existing file envelopes and sizes
  --overwrite           Safely regenerate existing VTI files

--skip-existing and --overwrite are mutually exclusive.
Outputs: <results-root>/<original_run_id>/vtis/step_*.vti

Examples:
  ./runConvertVTI.sh /mnt/rbc_hdf5 --jobs 4 --overwrite
  ./runConvertVTI.sh /mnt/rbc_hdf5 --jobs 4 --skip-existing --results-root results

Build settings use the existing Makefile/environment conventions. For example:
  RBC_ARGS='--case 3 --grid 512 --rr 3 --arch sm_89' ./runConvertVTI.sh /mnt/rbc_hdf5
Default build: RBC-1, 512^3, RR3, sm_86. CXX, CXXFLAGS, HDF5_FLAGS and PYTHON
can also be configured through the environment.
EOF
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi

make -C "$root" all
exec "$root/convertVTI" "$@"
