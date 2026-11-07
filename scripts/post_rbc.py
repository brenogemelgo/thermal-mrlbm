#!/usr/bin/env python3
"""Compatibility entry point; use python3 postRayleigh.py from the repository root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from postRayleigh import (
    KOOIJ, LABELS, NIEMELA, NUS, block_errors, clean_json, convergence,
    grid_convergence, load, main, metadata, profiles, report, save, scaling,
    uncertainty,
)


if __name__ == '__main__':
    main()
