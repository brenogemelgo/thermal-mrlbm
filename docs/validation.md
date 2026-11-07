# Focused validation — 2026-11-07

Tests ran locally with CUDA 13.0, `sm_86`, on an NVIDIA GeForce RTX 3050 Laptop
GPU (4 GiB). No production campaign or statistically converged turbulent run was
launched. Machine-readable results are in [validation-results.json](validation-results.json).

The complete suite passed in `/tmp/thermal-rbc-final-validation-v2`. Additional
checks subsequently covered hydrodynamic shear refinement, the latest checkpoint
reader/launcher, optional full-field export, and CUDA race checking. Reproduce the
suite with `python3 tests/validate_rbc.py` (it creates a fresh temporary directory).

| Check | Result |
| --- | --- |
| All 12 case/grid physical parameter sets | Ra, Pr and relaxation derivations pass |
| Random D3Q27 reconstruction | Mass, momentum and second moments conserved within FP32 tolerance |
| Force consistency | Independent MRT/Guo second moments and third-order source balance pass |
| D3Q27 degeneracies | Hxxx/Hyyy/Hzzz vanish; seven representable third-order components pass |
| RR2/RR3 limits | First/second moments agree; zero-velocity reconstruction and conduction profiles agree |
| NEE walls | Exact imposed velocity/temperature, extrapolated density/stress/thermal flux pass |
| Manufactured gradients | GPU kinetic/thermal dissipation and FP64 plane reductions match analytic discrete derivatives |
| Conduction, 12³, 200 steps, RR2 and RR3 | Largest absolute error over all five Nus and all samples: **3.80e-6** |
| Restart inside averaging | Checkpoint moments/accumulators, profiles, convergence CSV and time series are byte-identical to uninterrupted execution |
| Interrupted diagnostics | Rows newer than the committed checkpoint are removed on restart |
| Bad checkpoints | Legacy/incompatible magic and truncated state rejected |
| Short finite-Ra runs | RBC-1 at 64³ with RR2/RR3; RBC-2/3 at 16³ and RBC-4 at 32³ remain finite for 200 steps |
| Python analysis | Correlation-aware uncertainty checks, all plotting/report modes, PNG/PDF/JSON generation pass |
| CUDA memcheck | 0 errors on the short conduction integration |
| CUDA racecheck | 0 errors/warnings on forced moments, NEE and GPU dissipation reductions |
| Jet | Latest target compiles; turbulent jet behavior was not validated |
| Launcher | Build, short run and continuation pass without replacing the tracked executable |

## Wall refinement

For linear conduction plus a decaying temperature sine mode, and a simultaneous
horizontal shear sine mode between stationary plates, compare to the exact
heat/viscous diffusion solution after approximately 0.02 H²/kappa lattice steps.
Both fields are homogeneous in X/Y. FP64 state isolates truncation error from
long-integration FP32 roundoff; normal production state remains FP32.

| Grid | Max temperature error | Lower-wall thermal flux error | Max shear velocity error |
| --- | --- | --- | --- |
| 12³ | 1.68506e-4 | 3.70843e-4 | 3.34503e-5 |
| 24³ | 2.18649e-5 | 1.82585e-5 | 4.50130e-6 |
| 48³ | 2.77250e-6 | 4.51362e-6 | 5.83055e-7 |

Using the actual spacings 1/11, 1/23 and 1/47, temperature orders are 2.77 and
2.89; shear orders are 2.72 and 2.86. These manufactured cases support at least
second-order wall accuracy, not a general third-order claim. The finest wall-flux
refinement has order about 1.96. In the original FP32-only refinement the 48³
solution reached a roundoff floor (temperature error about 1.38e-5 and wall-flux
error about 3.02e-4); it was not used to infer truncation order.

## Numerical changes and modified files

- `src/constants.cuh`, `src/D3Q27.cuh`, `src/D3Q19.cuh`,
  `src/deviceFunctions.cuh`, `src/reconstruction.cuh`: unscaled SoA fields,
  non-equilibrium stress, physical/post-population velocity distinction,
  force-shifted second/third moments and RR2/RR3 specialization. Both orders
  retain fourteen persistent fields; no population arrays were added.
- `src/kernel.cuh`, `src/initialConditions.cuh`: normalized stress collision,
  quadratic force correction, density-independent buoyancy acceleration,
  physical velocity in thermal equilibrium, consistent initialized stress/flux.
- `src/boundary/deviceRuntime.cuh`, `src/boundary/jetIRBC.cuh`: RBC uses on-node,
  force-corrected NEE; legacy jet IRBC is isolated and adapted to unscaled
  reconstruction. Plate update follows bulk collision in the same CUDA stream.
- `cases/rayleighBenard.cuh`: all four families, actual H=NZ-1, prescribed Uff,
  viscosity/diffusivity/buoyancy derivation and free-fall schedules.
- `src/rbcStatistics.cuh`, `src/checkpoint.cuh`, `src/output.cuh`, `src/main.cu`:
  FP64 GPU reductions, five independent Nu formulas, profile/scalar accumulators,
  native versioned atomic checkpoints, restart checks and reduced default I/O.
- `scripts/run_rbc.py`, `postRayleigh.py`, `scripts/post_rbc.py` (compatibility),
  `postCommon.py`, `run.sh`,
  `runContinue.sh`: case selection, analysis/figures/reference comparisons,
  physical-velocity field reading, build isolation and output preservation.
- `tests/moments.cu`, `tests/validate_rbc.py`, `.gitignore`, `README.md`, `docs/*`:
  focused validation, documentation and generated-file exclusions.

The pre-existing modified `mrlbm` binary and deleted `postJet.py` /
`postRayleighBenard.py` files were left as found. The new implementation was built
under temporary paths or `build/`, preserving the user's existing executable.

## Unresolved physical validation and limits

The short buoyant integrations deliberately include startup transients and tiny
non-production grids. They do not establish steady heat transfer, agreement with
Niemela/Kooij, stability for the production duration, or DNS/grid convergence.
The comparison and convergence tools are implemented; production results are
still required to make those assessments.

On the 16³, Ra=1e6, 300-step startup test, mean density rose by about **0.145%**.
The 64³, 200-step RR3 test changed mean density by about **0.0063%**. NEE is not
an exactly mass-conserving closure; long-run density drift must be assessed along
with finite-Mach/acoustic startup effects. The time series records mean density
and peak speed. No unrequested mass renormalization was introduced.

The passive-temperature equilibrium remains first order in velocity, as in the
existing thermal architecture. RR3 extends the hydrodynamic Hermite expansion;
it does not supply the unavailable D3Q27 diagonal cubic modes or eliminate all
finite-Mach scalar errors. Dissipation diagnostics use second-order finite
differences and require grid-refinement evidence.

Old full-moment checkpoint files are deliberately incompatible. New checkpoints
preserve statistics exactly on the same ABI/endianness, but are not a portable,
checksummed interchange format. Production sizes 384³ and 512³ exceed this GPU's
available memory; only the listed short runs were executed.
