#!/usr/bin/env python3
"""Compatibility entry point for the compiled convertVTI application."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    binary = ROOT/'build/convertVTI'
    if not binary.exists():
        raise SystemExit('Build convertVTI first: python3 scripts/build_convert_vti.py')
    os.execv(str(binary), [str(binary), *sys.argv[1:]])
