# Turbulent Rayleigh–Bénard workflow

The solver retains CUDA pull streaming, compile-time D3Q27 specialization, and two
14-field SoA moment buffers. Populations exist only as local reconstruction values.
The existing jet IRBC implementation is isolated in `src/boundary/jetIRBC.cuh`;
RBC compiles no IRBC tables or inverses. Existing local changes to the binary and
deleted legacy post-processing scripts are not replaced.

## Building and short runs

From the repository root:

```bash
python3 scripts/run_rbc.py --list
python3 scripts/run_rbc.py --case 1 --grid 64 --rr 3 --build-only
python3 scripts/run_rbc.py --case 1 --grid 16 --test-grid --steps 300 \
  --transient-steps 20 --sample 10 --checkpoint-every 100 --run-id short_rr3
python3 scripts/run_rbc.py --case 1 --grid 12 --test-grid --conduction --steps 200 \
  --transient-steps 10 --sample 5 --run-id conduction
```

`--build-only` does not launch a simulation. Omitting it runs the chosen case.
Use `--continue` with the same run ID, grid, physics, RR order and sampling schedule.
An existing run with checkpoints cannot be overwritten by a fresh run. The native
executable accepts `--steps N` (absolute end step), `--checkpoint-every N`,
`--runId ID`, `--continue`, `--hdf5-root PATH`, and `--fields`. The default output contains reduced
statistics and checkpoints. `--fields` additionally exports HDF5 temperature and
physical velocity at the checkpoint cadence. Independent `--field-every` and
`--slice-every` intervals are preferable; see [the production guide](production-rbc.md).

The existing `run.sh`/`runContinue.sh` entry points remain, with `CUDA_ARCH` and
`NVCC_FLAGS` overrides. They now build under `build/` and preserve existing output.
For example `NAME=RB RUN_ID=example NVCC_FLAGS='-DRBC_CASE=2 -DRBC_N=128' ./run.sh`.
The Python launcher is preferable for the full case/grid selection.

## Cases, scales and plate location

| Case | Ra | Pr | Cubes | Transient / average (free-fall times) |
| --- | --- | --- | --- | --- |
| RBC-1 | 1e6 | 0.7 | 64, 96, 128, 512 | 100 / 150 |
| RBC-2 | 1e7 | 0.7 | 128, 192, 256, 512 | 100 / 200 |
| RBC-3 | 1e8 | 0.7 | 256, 384, 512 | 200 / 300 |
| RBC-4 | 1e8 | 1.0 | 256, 384, 512 | 200 / 300 |

Walls are **on lattice nodes** at z=0 and z=NZ-1: H=NZ-1. Periodic lengths are
Lx=NX and Ly=NY, so a cubic node array has Gamma=N/(N-1), slightly above one.
Temperatures are Th=1 and Tc=0; Uff=0.05, DeltaT=1, and

```
nu    = Uff H sqrt(Pr/Ra)
kappa = Uff H / sqrt(Ra Pr)
gbeta = Uff^2 / (DeltaT H)
tff   = H/Uff
tau_s = 1/2 + 3 nu; tau_T = 1/2 + 3 kappa
```

`gbeta*(T-Tref)` is acceleration, with Tref=0.5. The force density is rho times
this acceleration; it is not divided by rho again. Ra=0 disables buoyancy and
perturbations and uses nu=0.01, kappa=nu/Pr. Uff=0.05 remains a nominal time unit
in that verification case; it is not a physical buoyancy velocity when Ra=0.
The 512³ moment buffers alone require 14 GiB; reductions require additional memory.
All four production cases now use 512³; see [the production guide](production-rbc.md).
No multi-GPU support is added.

Compile-time overrides: `RBC_CASE`, `RBC_N`, `RBC_RA`, `RBC_PR`, `RBC_STEPS`,
`RBC_TRANSIENT_STEPS`, `RBC_SAMPLE`, `RBC_STAMP`, `RBC_PERTURBATION`, and
`REGULARIZATION_ORDER=2|3`. Optional `MRL_DOUBLE_PRECISION` changes solver state to
FP64 for numerical verification; ordinary builds use FP32 with FP64 reductions.

## Stored moments, forcing and recursive regularization

The first ten fields are rho, v=(vx,vy,vz), and six symmetric **non-equilibrium**
stresses Pij=A2_post,ij/rho-vi*vj. Thermal fields are T and the three unscaled
post-collision first moments J. There are no scaleI/scaleII/scaleIJ coefficients.
The output stress names are pxx, pyy, pzz, pxy, pxz, pyz.

For pre-collision population velocity q and acceleration a, physical velocity is
u=q+a/2. Stored post-collision velocity is v=q+a=u+a/2. Diagnostics and
`postCommon.readPackedField` return the physical velocity by subtracting a/2;
legacy packed moment exports retain the stored v convention; current physical
HDF5 fields and offline VTI conversion use physical u.
Acceleration is zero on the plate nodes, where v=u=0.

Collision is performed on the deviator and trace of the pre-collision stress P,
with their shear and bulk rates. The equal-rate form is

```
P_post = (1-omega) P_pre - omega a a / 4
```

The quadratic term is required when expressing time-centered Guo forcing in the
post-population velocity frame. This follows the shifted non-equilibrium storage
convention used in `pf-mrlbm`, adapted to density-normalized thermal hydrodynamics.

RR2 reconstructs equilibrium plus H2:P. RR3 also reconstructs the seven available
third-order Hermite components. Define C2_post=P_post+aa/4 and u=v-a/2. Then

```
A3_post/rho = u u u + sym(u C2_post) + sym(a u u)/2
```

Here `sym` denotes the three tensor placements, including their multiplicities.
This is a force-consistent recursion: C2_pre=P_pre+aa/4, A3_pre/rho=uuu+
sym(u C2_pre)-sym(auu)/2. Applying collision and the third-order Hermite force
source recovers the post expression. Hxxx=Hyyy=Hzzz=0 on D3Q27; these degenerate
modes are never reconstructed. There are six xxy-type modes with multiplicity
three and one xyz mode with multiplicity six. RR3 does not remove D3Q27's missing
diagonal cubic modes or all finite-Mach errors. The passive-temperature lattice
retains its established first-order equilibrium geq=w(T+3 c·Tu) and first-moment
regularization; it is not a fully compressible thermal-energy model.

## Plate NEE

After the bulk stream/collide kernel, a separate kernel reads **newly updated**
interior node 1 (or NZ-2) and writes every plate moment. It is the compact-moment
form of second-order, on-node non-equilibrium extrapolation:

```
f*_wall = feq(rho_1,0) + projection_H2(f*_1-feq(rho_1,u_1)-S_1/2)
g*_wall = geq(Twall,0) + (g*_1-geq(T_1,u_1))
```

Thus rho_wall=rho_1, v_wall=0, P_wall=P_1+aa/4, and
J_wall=J_1-T_1*u_1. Removing the half-force source avoids carrying spurious
momentum or forcing stress into the stationary wall. Wall reconstruction is
second order even in RR3 builds (u=a=0 there). All directions use the same
normal interior neighbor. Bulk pull streaming never addresses nodes outside
Z; X/Y wraps remain periodic. Plate moments are not collided a second time.
A conductive initialization includes the steady post-collision thermal flux
`Jz=-(tau_T-1) cs² dT/dz`, avoiding an initial heat-flux relaxation transient.
NEE fixes the wall macros exactly but is not an exact global mass-conserving
boundary algorithm; the mean density time series makes any accumulated drift visible.

## Diagnostics and restart

Sampling is at configurable lattice-step intervals; statistical accumulators only
include steps **strictly greater than** the transient. The first average sample is
`transient_steps+sample_stride`. Pre-transient samples remain in the time series
with `averaging=0`. Global integrals use a trapezoidal Z rule with half-weight
wall planes; horizontal means count the periodic points once.

Two GPU reduction stages return only O(NZ) FP64 plane data. Statistics include
means and raw second moments of T and each velocity component, physical
convective heat transport w(T-Tref), conductive heat flux -kappa*dT/dz, kinetic
energy, epsilon_u=2 nu S:S, and epsilon_T=kappa|grad T|². Derivatives are centered
second order, with one-sided second-order Z derivatives at plates. Profiles use
RMS about the combined plane/time mean. The local Kolmogorov estimate
eta=(nu³/epsilon_u)^(1/4) is computed **before** reduction; profiles report mean eta,
minimum eta over space/time and Delta/eta_min. Zero dissipation gives infinite eta.
This diagnostic is finite-difference based, not a claim that all dissipative
scales have been resolved.

The five separate estimates are

```
Nu_bottom,top = -H/DeltaT * <dT/dz>_plate
Nu_volume    = 1 + H/(kappa DeltaT) * <w(T-Tref)>_V
Nu_kinetic   = 1 + H^4 Pr²/(nu³ Ra) * <epsilon_u>_V
Nu_thermal   = H²/(kappa DeltaT²) * <epsilon_T>_V
```

The dissipation identities require stationary averaging; their instantaneous
values need not agree with the heat-flux estimates. In the Ra=0 conduction test,
Nu_kinetic is assigned its conduction limit 1 (the finite-Ra formula is singular).

Single-case output remains under `output/rayleigh_benard/RUN_ID/` by default.
The production driver relocates raw HDF5 via `--hdf5-root`; local run directories
contain a `hdf5` link for existing readers. Production history copies reside in
that HDF5 run directory under `checkpoint_history/`. Dataset contents:

- `metadata.h5:/metadata`: grid, physics, relaxation, storage/velocity convention and schedules.
- `statistics.h5:/statistics/timeseries`: five Nus, moments, RMS, energy/dissipation,
  Nu(z) range, layer thickness, minimum eta, maximum speed and mean density.
- `/statistics/profiles`: averaged plane profiles, fluxes, RMS, dissipation and eta.
- `/statistics/convergence`: Welford means/stddev and **uncorrelated** errors,
  labeled `naive_stderr`; `/statistics/runtime` records sustained throughput/ETA.
- `fields/step_*.h5` and `slices/step_*.h5`: independently scheduled physical fields.
- `checkpoints/step_*.chk`: unchanged versioned solver state, exact schedule,
  all FP64 accumulators, Kahan compensation, Welford state and sample counters.
- Closed HDF5 histories pair with checkpoints; they reside in `checkpoint_history/`
  under an external HDF5 root, or the legacy `checkpoints/` for co-located output.

Processed diagnostics, plots and reports use `results/<original_run_id>/`.
See [the production guide](production-rbc.md) for the complete HDF5 schema,
retention, transaction semantics, schedules and storage calculations.

Checkpoint writes use a temporary file and atomic rename. Restart rejects old
full-moment binaries, incompatible grids/physics/precision/RR/schedules and
truncated files. Newer time-series rows are rolled back to the committed step,
so interrupted output does not double-count samples. The averaging window can
be extended by choosing a later absolute end step. Checkpoint transfer is the
explicit full-field I/O exception; sampling never copies full fields to the host.
The version-2 format uses the native host ABI and endianness, not a portable archive.

## Post-processing and comparisons

Requires NumPy, Matplotlib and h5py:

```bash
python3 postRayleigh.py all output/rayleigh_benard/short_rr3 --out results
python3 postRayleigh.py scaling output/rayleigh_benard/rbc{1,2,3}_n*_rr3
python3 postRayleigh.py grid output/rayleigh_benard/rbc4_n*_rr3
```

`postRayleigh.py` is the main plotting script at the repository root. The previous
`scripts/post_rbc.py` command remains available as a compatibility entry point.

The script exports 300-dpi PNG and vector PDF figures, JSON numerical reports,
Nusselt histories/cumulative averages, a block-error ladder, RMS/dissipation
profiles, Nu(Ra) comparisons, and pairwise grid changes. Error estimates use an
initial-positive-pair autocorrelation estimate and nonoverlapping batch means.
A short record cannot establish stationarity or a block-error plateau. Reports
include first/second-half drift, all-estimator spread, averaged Nu(z) range,
separate plate boundary-layer cell/node counts, Delta/eta, and achieved averaging
duration. `dns_convergence_established` remains false; it requires external
assessment of time, grid and resolution diagnostics. No fitted scaling exponent
or Richardson order is inferred from insufficient or unmatched runs.

RBC-1–3 are compared to Nu=0.124 Ra^0.309 from
[Niemela et al., Nature 404 (2000)](https://doi.org/10.1038/35009036).
That experiment uses a helium cylinder with Gamma=0.5, unlike the horizontally
periodic Cartesian solver. The comparison is an experimental scaling reference,
not a geometry-matched acceptance value.

RBC-4 is compared to Nu=32.32 +/- 0.24 from Table 1 of
[Kooij et al., Computers & Fluids (2018)](https://doi.org/10.1016/j.compfluid.2018.01.010)
([author preprint](https://arxiv.org/abs/1802.09054)). Their periodic domain has
Gamma=1; the current node-array cubes approach that aspect ratio under refinement.
Their quoted interval summarizes the code/estimator comparison and does not
replace this simulation's sampling uncertainty.

Recursive Hermite reconstruction follows the framework in
[Malaspinas (2015)](https://arxiv.org/abs/1505.06900) and
[Coreixas et al. (2017)](https://arxiv.org/abs/1704.04413), with the explicit
force shifts above required by this solver's post-collision storage convention.

## Focused validation

```bash
python3 tests/validate_rbc.py --work /tmp/thermal-rbc-tests
compute-sanitizer --tool memcheck --error-exitcode 9 \
  build/rbc1_n12_rr3 --runId memcheck --steps 30
```

The suite checks all twelve production parameter derivations, randomized D3Q27
quadrature/moment conservation, second/third-order force-source identities,
Hermite degeneracies, RR2/RR3 limits, exact NEE macros and projected moments,
GPU gradient/dissipation reductions, thermal and hydrodynamic shear diffusion refinement (FP64 to
separate truncation error from FP32 roundoff), all five conduction Nus, split-run
bitwise restart including stale-row rollback, incompatible/truncated checkpoint
rejection, short finite-Ra case specializations and plot/report generation.
The jet target is compiled but not run as a turbulent jet validation.
