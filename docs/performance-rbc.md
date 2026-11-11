# Focused RBC kernel and statistics optimization

Reference: `~/Desktop/lagrava-mrlbm`, commit `50061a5`, especially
`src/kernel.cuh` and `src/refinement/utilities.cuh`. The comparison baseline is
an untouched copy of this thermal solver's pre-optimization working tree, not
its older Git HEAD. A stray trailing `5` in `src/output.cuh` prevented compilation;
it was removed in both baseline and optimized trees before comparison.

## Changes and synchronization boundaries

- `streamCollide` now writes the no-slip/isothermal NEE plates from the adjacent
  interior threads' freshly collided moments. Each interior node owns itself;
  z=1 also owns z=0, and z=NZ-2 owns z=NZ-1. Plate threads return without writing.
  NZ>=4 guarantees distinct donors. All pulls read the old, separate moment
  buffer. No thread reads another block's new output, and every output has one
  writer. This removes one timestep launch and fourteen donor-field reads per
  wall node while keeping the original post-collision boundary timing.
- `writeNEEPlate` shares the same second-order NEE projection with the standalone
  boundary kernel. It retains Pij+a_i*a_j/4 and j_T-T*(v-a/2), physical wall u=0,
  fixed T, force-free wall nodes, and H=NZ-1. It does not substitute the reference
  solver's previous-time donor for this thermal solver's current-time donor.
- Initialization retains its boundary pass: `initializeCase` writes every node,
  so adding concurrent plate writes there would create two writers per wall.
- Population pulls remain specialized over all 27 directions. The hot moment
  normalization, thermal collision and stores use `constexpr_for`; donor T is
  reused. Existing reconstruction/collision component loops already have explicit
  unroll directives. RR2/RR3 Hermite coefficients and forcing are unchanged.
- Statistics run only on the existing sample schedule (including the initial
  sample). Compile-time field/axis indices scalarize the gradients and moments;
  the Z temperature derivative is reused for conductive flux. Warp shuffles and
  transposed shared staging replace repeated shared-memory reduction passes.
  The tile stage uses one block-wide barrier; the plane stage uses none and
  processes eight independent observables per block. Positive fourth roots use
  `sqrt(sqrt(x))` instead of generic `pow(x,0.25)`.
- Device reduction buffers and the host time-series row are allocated once.
  Only 18*NZ FP64 profile values cross to the host per sample (72 KiB at NZ=512).
  CPU Kahan/Welford accumulation, HDF5 schema and checkpoint state stay intact.
  The progress path avoids a redundant device synchronization after a completed
  save; heartbeat-only reports still wait for GPU completion.
- Gradient statistics need neighbors from the fully updated field, and their
  plane reduction needs all tile outputs. Those grid-wide dependencies remain
  separate sampled launches. A normal block barrier cannot safely fuse them
  into stream/collide. The blocking profile copy remains necessary for the
  existing host accumulators, validation checks and output.

The fourteen-field SoA layout, coalesced X access, two moment buffers and
512-thread bulk launch geometry are preserved. No population buffers or new
full-field arrays are introduced. Code was organized manually around the
reference's four-space indentation, braces and deliberate statement groups.

## Measurements (2026-11-10)

RTX 3050 Laptop GPU (4 GiB), driver 580.173.02, CUDA 13.0.88, sm_86;
Nsight Systems 2025.3.2 and Nsight Compute 2025.4.1. Builds use C++20, O3,
restrict, relaxed constexpr, extended lambdas, line info and resource reporting.
Both variants use RBC-1, FP32/RR3, a 20-step transient and sampling every 100
steps. Each native-solver run advances 300 steps and writes one final checkpoint;
there are five repeats, alternating the order of before/after runs. Field
snapshots are disabled. The GPU-event harness warms up and takes the median of
five coupled-step batches; each reduction stage is averaged over 20 launches.

Native solver wall time includes both lattices, statistics, runtime records,
profile/metadata output and the final checkpoint. It excludes initialization,
module setup and the initial diagnostic sample. Final checkpoints are much more
frequent here than in production, so these absolute rates are not production
runtime predictions. The GPU/desktop and filesystem are shared; clocks are not
locked and temperatures vary. The overlapping ranges below do not establish a
repeatable end-to-end speedup.

| Grid | Before MLUPS, median [range] | After MLUPS, median [range] | Median change |
| --- | ---: | ---: | ---: |
| 64³ | 465.56 [432.47, 471.74] | 472.36 [455.83, 476.30] | +1.46% |
| 128³ | 408.03 [392.03, 421.81] | 404.19 [384.98, 420.57] | -0.94% |

Unprofiled CUDA event times, milliseconds. The coupled step includes the old
boundary launch in the baseline and fused boundary writes in the new kernel:

| Grid | Coupled step before / after | Tile stage before / after | Plane stage before / after | Total reduction improvement |
| --- | ---: | ---: | ---: | ---: |
| 64³ | 0.516164 / 0.521052 | 2.11251 / 1.72739 | 0.029338 / 0.011469 | 18.8% |
| 128³ | 4.952420 / 5.079640 | 13.89520 / 12.41270 | 0.048998 / 0.022467 | 10.8% |

These isolated reduction gains coexist with essentially unchanged end-to-end
rates. The sampled coupled harness also shows no overall improvement, so the
results do not justify a general speedup claim. Nsight Systems confirms removal
of the timestep NEE launch: for its 140-step harness workload, plate calls fall
from 141 to 1, retaining only initialization. Profiler replay timings are
recorded separately from the unprofiled event measurements.

### Registers, local memory and occupancy

| Kernel | Registers before / after | Local bytes/thread before / after | Shared bytes/block before / after |
| --- | ---: | ---: | ---: |
| streamCollide | 64 / 56 | 0 / 0 | 0 / 0 |
| reduceTiles | 48 / 56 | 144 / 0 | 2048 / 1152 |
| reducePlanes, 64³ | 24 / 15 | 0 / 0 | 2048 / 0 |
| reducePlanes, 128³ | 24 / 19 | 0 / 0 | 2048 / 0 |

PTXAS reports **zero spill stores and loads** for these kernels in both variants.
The old tile stage's 144-byte local array is explicit local storage, not a
reported register spill. Bulk theoretical occupancy remains 66.7%. Tile-stage
occupancy falls from 83.3% to 66.7% as its arrays become registers, while its
execution time improves. The small plane stage remains theoretically 100%.

Nsight Compute achieved occupancy at 64³ (before / after):

| Kernel | Achieved occupancy |
| --- | ---: |
| streamCollide | 61.45% / 61.54% |
| reduceTiles | 82.59% / 63.29% |
| reducePlanes | 92.98% / 75.61% |

A 32x4x1 bulk block was also measured. It raised theoretical occupancy to 83.3%
and used 48 registers, but coupled throughput fell to about 369 MLUPS at 64³
and 212 MLUPS at 128³. The default **32x4x4** geometry is retained because the
higher-occupancy alternative performed worse. No register cap or spill-inducing
launch bound was introduced.

Global allocation is unchanged. State plus reduction buffers require 29,516,800
bytes at 64³ and 236,079,104 bytes at 128³. The measured cudaMemGetInfo deltas
were respectively 32 MiB and 228 MiB in both variants, including allocation
rounding/runtime effects but excluding the preexisting context and desktop.
At 512³, two FP32 moment buffers alone require **14 GiB**, with about 72 MiB more
for reduction buffers, plus output slabs and CUDA overhead. Production grids
cannot run on this 4 GiB device and were not benchmarked.

## Numerical checks

- All fourteen checkpoint moment fields match the baseline **bitwise** after
  300 steps at both 64³ and 128³. Maximum differences in time-series diagnostics
  are 1.78e-15; averaged profiles differ by at most 1.78e-13, primarily from FP64
  sum reassociation and fourth-root rounding. No physical-state differences were
  observed in these cases.
- FP64 manufactured V2 (thermal advection/source) and V3 (time-dependent forced
  flow), 16³ for 80 steps, also have bitwise-identical moment fields. Diagnostic
  differences are at most 3.56e-15. These comparisons supplement independent
  moment/forcing and manufactured-reference tests; they do not establish DNS
  convergence.
- The existing full focused suite passed: moment quadrature, MRT/force and RR3
  consistency, Hermite degeneracies, NEE diffusion/shear refinement (12/24/48),
  pure conduction with all five Nu estimates and RR2/RR3 limits, bitwise restart
  plus history rollback/corruption rejection, short buoyant specializations,
  post-processing, and jet compilation.
- A poisoned-output test checks that fused streaming writes every moment and
  that an independent boundary pass on the new buffer changes no value. This
  tests donor time, all wall fields and ownership on block fringes, including
  the odd 7³ grid under memcheck.
- Compute Sanitizer memcheck, racecheck, synccheck and initcheck pass with no
  errors or warnings on the final 12³ moment/boundary/gradient workload. Unique
  global writers were also checked by inspection; racecheck alone does not
  prove global-memory race freedom.

The final benchmark repeat encountered a full filesystem during checkpoint
output. Identical task-generated checkpoints were verified and hard-linked,
reclaiming about 3.5 GB without changing their contents. The interrupted pair
was rerun successfully; failed output remains separate. A free-space check now
prevents starting the benchmark when checkpoint payload plus build margin will
not fit. No production data was changed or deleted.

## Reproduce and inspect

The machine-readable report, compiler logs, sanitizer logs, profiling summaries,
individual wall-time logs and before/after source snapshots are in
`results/performance_20261110/`. The report is `report.json`; raw short-run
artifacts remain in `/tmp/thermal-rbc-perf-artifacts/` for this session.

```bash
# Current implementation only; bounded to short non-production workloads.
python3 scripts/benchmark_rbc.py --work /tmp/rbc-current-benchmark

# Before/after, using the preserved original source snapshot.
python3 scripts/benchmark_rbc.py \
  --before results/performance_20261110/source-before \
  --work /tmp/rbc-comparison --steps 300 --repeats 5

# Existing short correctness suite, including fused plate ownership/fringes.
python3 tests/validate_rbc.py --work /tmp/rbc-correctness

# Profiler example after building a short harness; use a fresh working directory.
nsys profile --trace=cuda --sample=none --cpuctxsw=none ./bench 20
ncu --kernel-name 'regex:reduceTiles' --launch-count 1 --set basic ./bench 20
```

`--work` must be new/empty; completed results are preserved. The benchmark CLI
limits grids to 32/64/96/128 and timesteps to 1000. It compares final moments
exactly and permits only diagnostic roundoff (rtol=1e-11, atol=1e-12), which is
an implementation-equivalence allowance rather than a physical or DNS criterion.
It does not render video or invoke the production launcher.

Modified solver files: `src/kernel.cuh`, `src/boundary/deviceRuntime.cuh`,
`src/rbcStatistics.cuh`, `src/main.cu`, and the trailing-character correction in
`src/output.cuh`. Verification additions: `tests/benchmark_rbc.cu`,
`scripts/benchmark_rbc.py`, `tests/moments.cu`, and `tests/validate_rbc.py`.
Production case physics, output schedules and checkpoint layout are unchanged.
