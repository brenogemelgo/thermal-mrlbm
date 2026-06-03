#!/usr/bin/env bash
set -euo pipefail

CASE_NAME="${NAME:-}"
runId="${RUN_ID:-}"
extra_args=()

for arg in "$@"; do
    if [[ "$arg" == NAME=* ]]; then
        CASE_NAME="${arg#NAME=}"
    elif [[ "$arg" == RUN_ID=* ]]; then
        runId="${arg#RUN_ID=}"
    elif [[ "$arg" == RUNID=* ]]; then
        runId="${arg#RUNID=}"
    elif [[ "$arg" != --* ]]; then
        ARG_CASE_NAME="$(printf '%s' "$arg" | tr '[:lower:]' '[:upper:]')"
        if [[ -z "$CASE_NAME" && ( "$ARG_CASE_NAME" == "FOURIER" || "$ARG_CASE_NAME" == "CAVITY" || "$ARG_CASE_NAME" == "SLOT" || "$ARG_CASE_NAME" == "JET" ) ]]; then
            CASE_NAME="$ARG_CASE_NAME"
        elif [[ -z "$runId" ]]; then
            runId="$arg"
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
    SLOT)
        CASE_DEFINE="-DCASE_SLOT"
        ;;
    JET)
        CASE_DEFINE="-DCASE_JET"
        ;;
    *)
        echo "Unknown NAME=$CASE_NAME. Use NAME=FOURIER, NAME=CAVITY, NAME=SLOT, or NAME=JET." >&2
        exit 1
        ;;
esac

case_lower="$(printf '%s' "$CASE_NAME" | tr '[:upper:]' '[:lower:]')"

if [[ -z "$runId" ]]; then
    runId="000"
elif [[ "$runId" =~ ^[0-9]+$ ]]; then
    runId="$(printf '%03d' "$((10#$runId))")"
fi

if [[ -z "$runId" || "$runId" == "." || "$runId" == ".." || "$runId" == *"/"* || "$runId" == *"\\"* ]]; then
    echo "Invalid run id: $runId" >&2
    exit 1
fi

rm -rf "output/${case_lower}/${runId}"
mkdir -p "output/${case_lower}/${runId}"
rm -f mrlbm

nvcc -std=c++20 -O3 --restrict --expt-relaxed-constexpr --fmad=true --extra-device-vectorization --extended-lambda -arch=sm_86 -lineinfo -Xptxas -v "${CASE_DEFINE}" src/main.cu -o mrlbm
./mrlbm --runId "$runId" "${extra_args[@]}"
