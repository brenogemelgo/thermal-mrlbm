#!/usr/bin/env python3
"""Build the independent CPU HDF5-to-VTI application; never start a solver."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def hdf5_flags():
    if os.environ.get('HDF5_FLAGS'):
        return shlex.split(os.environ['HDF5_FLAGS'])
    try:
        return shlex.split(subprocess.check_output(['pkg-config', '--cflags', '--libs', 'hdf5'], text=True))
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError('Install HDF5 development files and pkg-config, or set HDF5_FLAGS') from error


def build_converter(flags=None):
    destination = ROOT/'build/convertVTI'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'convertVTI.{os.getpid()}.tmp')
    command = [*shlex.split(os.environ.get('CXX', 'c++')), '-std=c++20',
               *shlex.split(os.environ.get('CXXFLAGS', '-O3 -Wall -Wextra -Wpedantic')),
               str(ROOT/'tools/convertVTI.cpp'), '-o', str(temporary),
               *(hdf5_flags() if flags is None else flags)]
    print(shlex.join(command), flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    build_converter()


if __name__ == '__main__':
    main()
