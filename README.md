# thermal-mrlbm

CUDA moment-reconstruction LBM for coupled hydrodynamics and passive temperature,
with compile-time D3Q27 RR2/RR3, compact non-equilibrium stress storage, and
horizontally periodic Rayleigh–Bénard convection.

See [the RBC implementation and run guide](docs/rbc.md) for the four case families,
plate/forcing conventions, diagnostics, checkpoints, and post-processing.
[Validation results](docs/validation.md) describe focused tests and remaining limits.
The separate [pre-production campaign](docs/preproduction-validation.md) implements
V1–V5, manufactured solutions, sequential execution/resume, and consolidated
analysis. That campaign has not been compiled or run.

```bash
# Inspect the production configurations without running them.
python3 scripts/run_rbc.py --list

# Short verification only.
python3 scripts/run_rbc.py --case 1 --grid 16 --test-grid --steps 300 \
  --transient-steps 20 --sample 10 --run-id short
python3 postRayleigh.py all output/rayleigh_benard/short

# Focused CUDA tests, including restart and boundary refinement.
python3 tests/validate_rbc.py
```

Requires a CUDA C++20 toolchain; plotting/validation also uses NumPy and Matplotlib.
The launcher defaults to `sm_86`; set `--arch` (or `CUDA_ARCH` for the test suite)
for another GPU. Production DNS convergence must be established using time,
grid, dissipation, heat-flux and boundary-layer diagnostics.
