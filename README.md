# thermal-mrlbm

CUDA moment-reconstruction LBM for coupled hydrodynamics and passive temperature,
with compile-time D3Q27 RR2/RR3, compact non-equilibrium stress storage, and
horizontally periodic Rayleigh–Bénard convection.

See [the RBC implementation and run guide](docs/rbc.md) for the four case families,
plate/forcing conventions, diagnostics, checkpoints, and post-processing.
[Validation results](docs/validation.md) describe focused tests and remaining limits.
The separate [pre-production campaign](docs/preproduction-validation.md) implements
V1–V5, manufactured solutions, sequential execution/resume, and consolidated
analysis. See [the production campaign and HDF5 guide](docs/production-rbc.md) for
the four 512³ RR3 runs, output schedules, storage estimates and RBC-3-only video
post-processing. Full exports contain temperature and physical velocity; VTI is off.
See [the two-GPU PCIe guide](docs/multi-gpu.md) for overlapped peer halo exchange,
slab specialization, memory requirements and verification limits.

```bash
# Inspect the production campaign without compiling or running.
./runRBCProduction.sh --plan

# Inspect the same campaign configured for two PCIe GPUs.
./runRBCProduction.sh --gpus 2 --devices 0,1 --plan

# Launch manually when ready (runs the full campaign).
# ./runRBCProduction.sh

# Short verification only.
python3 scripts/run_rbc.py --case 1 --grid 16 --test-grid --steps 300 \
  --transient-steps 20 --sample 10 --run-id short
python3 postRayleigh.py all output/rayleigh_benard/short

# Focused CUDA tests, including restart and boundary refinement.
python3 tests/validate_rbc.py
```

Requires a CUDA C++20 toolchain, HDF5 C development files and pkg-config;
plotting/validation uses NumPy, Matplotlib and h5py. Movies also require ffmpeg.
Production HDF5 data lives in `output_hdf5/rayleigh_benard/<run_id>/` by default;
set `./runRBCProduction.sh --hdf5-root /path/to/storage` to relocate it. Binary
checkpoints remain under `output/rayleigh_benard/<run_id>/`; results use
`results/<run_id>/`. RBC-1/2 export fields every 5,000 steps; RBC-3/4 every 10,000.
All cases checkpoint every 100,000 steps. Builds also compile `build/convertVTI`: explicit offline
conversion uses `./build/convertVTI <hdf5-location> --jobs 4` (parallel CPU workers).
The launcher defaults to `sm_86`; set `--arch` (or `CUDA_ARCH` for the test suite)
for another GPU. Production DNS convergence must be established using time,
grid, dissipation, heat-flux and boundary-layer diagnostics.

The [focused performance report](docs/performance-rbc.md) documents fused NEE
processing, GPU reduction changes, before/after measurements and short-test results.
