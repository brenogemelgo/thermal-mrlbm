#!/usr/bin/env bash
set -euo pipefail

case_id="default"
if (($# > 0)) && [[ "$1" != --* ]]; then
    case_id="$1"
    shift
fi

if [[ -z "$case_id" || "$case_id" == "." || "$case_id" == ".." || "$case_id" == *"/"* || "$case_id" == *"\\"* ]]; then
    echo "Invalid case id: $case_id" >&2
    exit 1
fi

rm -rf "output/$case_id"
mkdir -p "output/$case_id"
rm -f mrlbm

nvcc -std=c++20 -O3 --restrict --expt-relaxed-constexpr --fmad=true --extra-device-vectorization --extended-lambda -arch=sm_86  -lineinfo -Xptxas -v src/main.cu -o mrlbm
./mrlbm --case-id "$case_id" "$@"
