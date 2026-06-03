#!/usr/bin/env bash
set -euo pipefail

CASE_NAME="${NAME:-}"
extra_args=()

for arg in "$@"; do
    if [[ "$arg" == NAME=* ]]; then
        CASE_NAME="${arg#NAME=}"
    elif [[ "$arg" != --* ]]; then
        ARG_CASE_NAME="$(printf '%s' "$arg" | tr '[:lower:]' '[:upper:]')"
        if [[ -z "$CASE_NAME" && ( "$ARG_CASE_NAME" == "FOURIER" || "$ARG_CASE_NAME" == "CAVITY" || "$ARG_CASE_NAME" == "JET" ) ]]; then
            CASE_NAME="$ARG_CASE_NAME"
        else
            extra_args+=("$arg")
        fi
    else
        extra_args+=("$arg")
    fi
done

if [[ -z "$CASE_NAME" ]]; then
    CASE_NAME="JET"
else
    CASE_NAME="$(printf '%s' "$CASE_NAME" | tr '[:lower:]' '[:upper:]')"
fi

case "$CASE_NAME" in
    FOURIER)
        CASE_DEFINE="-DCASE_FOURIER"
        ;;
    CAVITY)
        CASE_DEFINE="-DCASE_CAVITY"
        ;;
    JET)
        CASE_DEFINE="-DCASE_JET"
        ;;
    *)
        echo "Unknown NAME=$CASE_NAME. Use NAME=FOURIER, NAME=CAVITY, or NAME=JET." >&2
        exit 1
        ;;
esac

case_lower="$(printf '%s' "$CASE_NAME" | tr '[:upper:]' '[:lower:]')"

rm -rf "output/${case_lower}/000"
mkdir -p "output/${case_lower}/000"
rm -f mrlbm

nvcc -std=c++20 -O3 --restrict --expt-relaxed-constexpr --fmad=true --extra-device-vectorization --extended-lambda -arch=sm_86 -lineinfo -Xptxas -v "${CASE_DEFINE}" src/main.cu -o mrlbm
./mrlbm "${extra_args[@]}"
