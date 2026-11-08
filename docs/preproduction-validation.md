# Pre-production RBC numerical validation

**Implementation only: this campaign has not been compiled, executed or validated.**
No result or pass status is pre-populated. Run the following manually from the
repository root when ready:

```bash
./runRBCValidation.sh
```

This launches V1–V5 sequentially, with RR3, at every grid listed below. It builds
through `scripts/run_rbc.py`, runs the CUDA solver, then immediately invokes
`postRayleigh.py validation` for that run. Only after successful post-processing
is the next simulation started. The campaign can take substantial time: V4/V5
use the full existing transient/averaging schedules. Checkpoints are retained;
there is no automatic deletion or disk-space reclamation.

## Cases

| Test | Grids | Physics | Precision | Duration |
| --- | --- | --- | --- | --- |
| V1 | 16, 32, 64, 128 | Ra=0, u=0, T=1-Z | FP64 | H² lattice steps |
| V2 | 32, 64, 128 | Manufactured thermal transport; prescribed shear varying in Z and time | FP64 | 5 H² steps |
| V3 | 32, 64, 128 | Manufactured forced unsteady channel | FP64 | 5 H² steps |
| V4 | 64, 128 | Ra=1e6, Pr=0.7, Uff=0.05 | FP32 | 100 transient + 150 averaging tff |
| V5 | 128, 256 | Ra=1e7, Pr=0.7, Uff=0.05 | FP32 | 100 transient + 200 averaging tff |

All grids are N³. Plate nodes are z=0 and z=N-1; H=N-1. X/Y remain periodic,
with periodic length N, and the plates remain no-slip/isothermal with Th=1,
Tc=0. No population arrays are added: both lattices retain their existing
14-field SoA storage. The analytical cases use nu=kappa=0.02, Pr=1, no buoyancy.
Their diffusive time is s=kappa*t/H² (or nu*t/H² for V3); final s is 0.02 for
V1 and 0.1 for V2/V3. Their common normalized physical grid spacing is h=1/H.

X=x/N and Z=z/H are used below. X has coordinate spacing 1/N, while physical
x/H has spacing 1/H and period Gamma=N/H. The references use this actual
geometry at each resolution. Gamma changes slightly with N; refinement is a
family of on-node cubes approaching Gamma=1, rather than exactly identical
aspect ratios. Convergence is calculated using physical h=1/(N-1), never 1/N.

### V1: stationary conduction

```
T(x,z,t) = 1-Z
u = 0
Q = 0
```

Here `u` denotes velocity. All hydrodynamic velocity components start at zero.
The conductive post-collision thermal flux is initialized consistently as
Jz=-(tau_T-1) cs² dT/dz. The exact continuum temperature is also exactly linear
on the grid, so there is no nonzero second-order interpolation error to fit.
The report shows temperature errors, five Nu diagnostics, and both wall-flux
errors. Any order fitted to roundoff-level errors is explicitly inconclusive.
For Ra=0 the kinetic-dissipation identity is singular; the recorded kinetic Nu
is the existing conduction-limit value 1, not an independent finite-Ra test.

### V2: manufactured advection–diffusion

Let A=0.1, Pe=1, kx=2*pi/N and kz=pi/H. Prescribe

```
theta = A exp(-s) sin(2*pi*X) sin(pi*Z)
T = 1-Z+theta
u_x = (kappa/H) sin(pi*Z) cos(2*pi*s); u_y=u_z=0
Q = dT/dt + u_x*dT/dx - kappa*Laplacian(T)
  = [kappa*(kx²+kz²)-kappa/H²]*theta + u_x*dT/dx
```

This divergence-free velocity has both spatial and temporal dependence and
vanishes at the plates. The source is nonzero and includes the velocity-dependent
advection explicitly. The scalar is advanced numerically: its interior T is
never reset to the reference solution. Hydrodynamic moments are assigned the
prescribed velocity for this isolated scalar test, rather than claiming the
prescribed field is a freely evolving NS solution.

The thermal source is integrated using a transformed population convention.
Persistent TEMP is physical T. Reconstruction uses post-source zeroth moment
T+Q(t)/2; after streaming T=sum(g_pre)+Q(t+1)/2. Collision uses physical T and
the prescribed velocity at the new time. The source has population projection
w_i Q, with zero first moment; no unrequested Qu first-moment source is added.
NEE extrapolates the non-equilibrium first moment and sets the wall zeroth
moment explicitly, so the interior Q/2 shift is not copied into wall temperature.
The thermal equilibrium remains the solver's first-order scalar equilibrium.
Leading Chapman–Enskog fluxes initialize the manufactured scalar to reduce a
kinetic startup layer; the initial temperature itself is exact.

Post-processing independently evaluates L1/L2/Linf errors and recovers diffusivity
from consecutive numerical temperature slices by the least-squares PDE balance

```
kappa_recovered = sum(Laplacian(T_mid) * [delta_t T + u_mid*d_x T_mid - Q_mid])
                  / sum(Laplacian(T_mid)^2)
```

The time pair is exactly one lattice step apart, regardless of output interval.
Only interior Z nodes enter this balance; periodic centered X differences and
centered Z second differences are used. The report also evaluates the residual
with the advection term deliberately omitted and records the nonzero reference
advection norm. Thus conduction alone cannot satisfy the manufactured balance.
This recovery is a diagnostic with its own second-order finite-difference error,
not an exact finite-grid acceptance identity.

### V3: manufactured forced channel

With U0=0.2*nu/H, s=nu*t/H², f(s)=1+0.25*sin(2*pi*s), use

```
u_x = U0 f(s) sin(pi*Z); u_y=u_z=0
p = constant; rho_reference=1
F_x/rho = (nu*U0/H²) [f'(s)+pi²*f(s)] sin(pi*Z)
F_y=F_z=0
sigma_xz = sigma_zx = -nu*(U0/H)*pi*f(s)*cos(pi*Z)
```

The convective acceleration is zero because the flow is homogeneous in X/Y.
The applied force includes both the time derivative and viscous balance and
is **not** an unforced vortex. Density/velocity evolve through the existing
hydrodynamic kernels; interior velocities are never set to the exact solution.
T remains the conductive profile and has no feedback on this imposed force.
Diffusive scaling makes the lattice Mach number shrink under refinement.

Every consumer of force now receives the time of its moment buffer: old-time
force for population reconstruction, new-time force for collision and NEE,
and sample-time force for physical-velocity diagnostics. Stored v=u+a/2 and
P=A2_post/rho-vv retain the production convention. Initial stress uses the
force-corrected post-collision Chapman–Enskog value. The reference continuum
viscous stress is compared against

```
sigma = rho * (1-omega/2)/(1-omega) * (P + aa/4)
```

For nu=0.02, omega differs from one, so this conversion is nonsingular. Stress
is measured in its physical normalization nu*U0/H. Here sigma is the non-equilibrium viscous
momentum-flux stress, -2*rho*nu*S; Cauchy viscous stress has the opposite sign. All six independent tensor
components contribute to the error; the exact shear components are as above.

The Python reference also reconstructs the old state with independent Cartesian
D3Q27 quadrature, pull-streams it, and verifies the direct Guo second-moment and
momentum identities against the new state. It does not invoke CUDA reconstruction
functions. A conservative polynomial forward-error budget gamma_65536 times a
data-conditioning scale bounds this algebraic residual in FP64; it is not a
continuum NS error tolerance. A separate momentum PDE residual and velocity/
stress error convergence expose truncation errors that the algebraic identity
cannot detect.

## Execution and restart

```bash
./runRBCValidation.sh --only V1,V2,V3
./runRBCValidation.sh --skip V1 V2
./runRBCValidation.sh --only V3 --rr both
./runRBCValidation.sh --resume
./runRBCValidation.sh --post-only
./runRBCValidation.sh --root /path/to/campaign --arch sm_86
```

- `--only` / `--skip` accept comma-separated names or multiple names.
- `--rr 3` is the default. `--rr 2` selects RR2 for V3; `--rr both` runs both
  orders for V3. V1/V2/V4/V5 retain RR3. Repeat the same order selection on resume.
- `--resume` skips completed simulation work and post-processes its output;
  interrupted simulations rebuild through the existing launcher and load the
  latest committed checkpoint. A partial attempt without a checkpoint restarts
  from its initial condition only when `--resume` is explicitly supplied.
- `--post-only` neither builds nor runs simulations. It requires a successful
  simulation receipt for each selected job and regenerates its analysis.
- `--checkpoint-every STEPS` overrides all checkpoint intervals. Defaults are
  approximately four checkpoints per analytical run and every 10 tff for RBC.
- Run IDs are deterministic, for example `validation1_v3_n064_rr3_double`.
  Each test has a separate output tree. A manifest mismatch or an unowned output
  directory stops the campaign. Use another `--root` for changed settings.
- A fresh invocation refuses existing owned runs rather than overwriting them.
  A file lock prevents simultaneous drivers from mutating one campaign tree.
- A completed simulation receipt is committed only after a successful solver
  return and the final checkpoint. It survives post-processing failures.
  Per-run logs append new attempts. The driver stops on build, simulation,
  post-processing, or failed numerical checks, and retains all successful output.
- Analytical snapshot JSON manifests commit their corresponding binary slices.
  Post-processing ignores snapshots beyond the target end step; partial snapshots
  have no committed manifest. Restart replaces rerun diagnostic samples at the
  same step, with the existing statistics/checkpoint rollback behavior preserved.

Do not run two native binaries writing the same run ID outside the campaign lock.
Production checkpoint ABI is preserved. A validation-physics tag in the existing
header field marker prevents cross-loading V1/V2/V3 or production checkpoints.
The unused validation buoyancy/gravity fingerprint slots record amplitude, Pe,
channel Re and formula version, preventing a changed manufactured drive from
resuming incompatible accumulators.

## Output and numerical assessments

Default root: `output/rbc_validation/`.

```
V1/rayleigh_benard/validation1_v1_n016_rr3_double/
    validation_run.json          # immutable expected configuration
    simulation_complete.json    # successful solver receipt
    metadata.txt
    checkpoints/step_*.chk
    rbc_timeseries.csv, rbc_profiles.csv, rbc_convergence.csv
    validation/snapshot_STEP.{bin,json}
    execution.log, postprocess.log
V1/post/validation1_v1_n016_rr3_double/
    validation_result.json, checks.csv
    analytical_history.csv, analytical_profiles.csv
    conduction_nusselt.csv
    *.pdf, *.png
summary/
    campaign_summary.{json,csv,md}
    campaign_checks.csv
    spatial_convergence.csv
    rbc_grid_differences.csv
    rr2_rr3_comparison.csv
    *.pdf, *.png
campaign_state.json
```

V2 additionally plots recovered diffusivity and advection-sensitive PDE
residuals. V3 plots velocity, viscous stress and force consistency. Each
analytical snapshot contains only the XZ slice of the two existing moment buffers
plus a GPU-computed maximum variation over Y. The analytical problem is Y-invariant;
post-processing requires exact discrete Y invariance before treating slice norms
as full volume norms. Slice storage is O(N²), with no persistent populations or
additional O(N³) fields. Norm integration uses periodic X and half-weight wall
planes. Full-field export is not enabled by the campaign.

V4/V5 reuse all existing `postRayleigh.py` convergence/profile plots and numerical
reports. They add `nusselt_confidence.csv`, conditional 95% batch-bootstrap
intervals, and `nusselt_bootstrap.npz`. Block length is at least twice the estimated
integral correlation time, with at least eight complete blocks before an interval
is produced. The incomplete trailing block is omitted for variance estimation;
bootstrap draws are centered on the full-record mean. A fixed per-run/estimator hash seeds resampling reproducibly. These
are conditional uncertainty estimates, not proof of stationarity. Short or strongly
correlated records produce missing intervals and an inconclusive assessment.

The report includes five Nus, Nu(z) uniformity, mean/RMS velocity and temperature,
kinetic/thermal dissipation, boundary-layer cell/node counts, local Delta/eta,
record duration and half-record drift. Grid differences use independent-run
bootstrap distributions where available. No arbitrary exact RBC Nusselt target
is used. Agreement with a correlation, Nu alone, or Delta/eta alone never sets a
DNS-converged flag.

Checks are separated into **passed**, **failed**, and **inconclusive**. Exact
boundary assignments, finite state, prescribed-velocity consistency, local
force algebra, configuration and sample-count integrity have justified checks.
Continuum truncation error is reported as measured error/order, without an
invented finite-grid tolerance. Stationarity and DNS resolution remain inconclusive
until assessed from sufficient time/grid evidence. Unexecuted jobs remain
inconclusive; execution failures remain failed. The consolidated report includes
both run-level outcomes and individual check counts. It is regenerated on normal
completion and attempted even after interruption/failure. If summary generation
itself fails, `summary_failure.json` retains the error and campaign-state path.

Standalone analysis, without rebuilding or rerunning a simulation:

```bash
python3 postRayleigh.py validation \
  output/rbc_validation/V3/rayleigh_benard/validation1_v3_n064_rr3_double \
  --out output/rbc_validation/V3/post
python3 postRayleigh.py validation-summary output/rbc_validation \
  --out output/rbc_validation/summary
```

Python analysis requires NumPy and Matplotlib. No Python reference imports solver
physics or numerical-kernel code. The source and execution interfaces have only
been inspected as text for this implementation; no compilation, test, campaign,
simulation or post-processing invocation was performed.

## Source inventory

New files:

- `runRBCValidation.sh`: explicit executable entry point.
- `cases/rbcValidation.cuh`: V1–V3 physical constants and specialization.
- `src/validation/physics.cuh`: manufactured driving and prescribed velocity.
- `src/validation/output.cuh`: compact two-time-level slice output and Y check.
- `scripts/rbc_validation_cases.py`: the complete V1–V5 matrix and schedules.
- `scripts/rbc_validation_campaign.py`: execution, ownership, resume and receipts.
- `scripts/rbc_validation_reference.py`: independent continuum/quadrature references.
- `scripts/rbc_validation_analysis.py`: per-run analysis and campaign summaries.
- `docs/preproduction-validation.md`: assumptions, equations and manual workflow.

Updated files:

- `scripts/run_rbc.py`: validation selection, precision and output-root options.
- `postRayleigh.py`: `validation` and `validation-summary` analysis modes.
- `postCommon.py`: force-shifted V3 full-field velocity reading, if requested.
- `src/constants.cuh`, `src/initialConditions.cuh`, `src/reconstruction.cuh`,
  `src/kernel.cuh`, `src/boundary/deviceRuntime.cuh`, `src/rbcStatistics.cuh`:
  validation-only physics and buffer-time-aware force/source consumers.
- `src/main.cu`, `src/output.cuh`: optional output root, slice hooks and metadata.
- `src/checkpoint.cuh`: manufactured-physics identity in checkpoint validation.
- `README.md`: campaign documentation link.
