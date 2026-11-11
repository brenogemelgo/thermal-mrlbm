# Production RBC: storage, HDF5 location and offline conversion

All four production grids are **512³**, with RR3, `Uff=0.05`, on-node plates
`H=511` and `tff=10220` lattice steps. Temperature is 1/0 at the plates and X/Y
remain periodic. The numerical kernels, force convention, compact 14-field SoA
storage and FP64 statistics are unchanged by the output infrastructure.

## Final campaign defaults

| Case / original run ID | Ra | Pr | Transient steps | Averaging steps | Final step |
| --- | ---: | ---: | ---: | ---: | ---: |
| RBC-1 / `rbc1_n512_rr3` | 1e6 | 0.7 | 1022000 | 1533000 | 2555000 |
| RBC-2 / `rbc2_n512_rr3` | 1e7 | 0.7 | 1022000 | 2044000 | 3066000 |
| RBC-3 / `rbc3_n512_rr3` | 1e8 | 0.7 | 2044000 | 3066000 | 5110000 |
| RBC-4 / `rbc4_n512_rr3` | 1e8 | 1.0 | 2044000 | 3066000 | 5110000 |

`nu=Uff H sqrt(Pr/Ra)`, `kappa=nu/Pr`, `g beta=Uff²/(H DeltaT)`,
`tau=0.5+3nu`, `tau_T=0.5+3kappa`. These scales use the actual wall separation.
The horizontal aspect ratio is `512/511`; compiled FP32 constants have roundoff.

- **3D fields: every 5000 steps for RBC-1/2 and 10000 steps for RBC-3/4**, plus
  initialization and the final step. These intervals are approximately 0.489
  and 0.978 tff, respectively. Each file contains only FP32 temperature and
  physical ux/uy/uz.
- **Checkpoints: every 100000 steps**, plus completion. Keep the latest two
  checkpoint/history pairs per case. Checkpoints do not include step zero.
- **Statistics: every 100 steps**, independently of field/checkpoint output.
  The existing post-transient GPU reductions, Kahan profile sums and Welford
  accumulators are unchanged.
- **Average profiles: every 10220 steps (1 tff)** and checkpoints/completion.
  These are the latest cumulative averages, not a growing full-field history.
- **Progress heartbeat: every 1022 steps (0.1 tff)**, and every save/sample point.
- Separate 2D slices are off; `--slice-every` can enable them explicitly.
- Python video preview is enabled only for RBC-3, unless `--no-video` is supplied.
  It reads an XZ hyperslab of each volume when separate slices are absent. This
  is a 2D preview; actual 3D volume rendering is an offline visualization step.
- **The simulation never produces VTI.** Conversion is a separate compiled CLI,
  never launched by the simulation or production driver.

Intervals are global lattice-step multiples. Initial/final field counts include
an extra final file when the end step is off cadence. `--output-start` gates
fields/slices only. Changing field cadence leaves the online Nu, RMS,
dissipation, profiles and statistical-convergence calculations unchanged.
Potential additional full-field analyses include horizontal spectra, spatial
correlations, plume/vortex maps and conditional boundary-layer statistics;
these are not implemented in `postRayleigh.py` and need adequate offline samples.
A coarser checkpoint cadence increases the amount of work lost after interruption.

## Separate HDF5 destination

The production option is **`--hdf5-root DIRECTORY`**, defaulting to
`output_hdf5/` in the repository. It may be an absolute path on another mounted
filesystem. The destination is organized by the unchanged original run IDs:

```
<HDF5_ROOT>/rayleigh_benard/<run_id>/
    metadata.h5
    statistics.h5
    fields/step_*.h5
    slices/step_*.h5                # only when explicitly enabled
    checkpoint_history/statistics_step_*.h5

output/rayleigh_benard/<run_id>/
    production_run.json, simulation_complete.json
    checkpoints/step_*.chk          # binary restart data remains here
    execution.log
    hdf5 -> <HDF5_ROOT>/rayleigh_benard/<run_id>/

results/<run_id>/
    report.json, diagnostics.h5, *.pdf, *.png
    temperature_plumes.mp4, video.json  # RBC-3 preview only
    vtis/step_*.vti                     # explicit convertVTI only
    postprocess.log, production_post_complete.json
```

`--output-root` relocates local run records/binary checkpoints; `--results-root`
relocates processed results. The per-run `hdf5` directory symlink lets existing
post-processing commands find relocated data. Readers accept either the local
run directory or its actual HDF5 run directory, preserving the original ID.
The location is recorded in the immutable production configuration; repeat the
same options on resume/post-only. Conflicting data links or unowned external
output stop with an error. No old output is moved, deleted or adopted implicitly.
Mount the same destination before restarting. Checkpoint history HDF5 files live
on the selected HDF5 filesystem, while the authoritative `.chk` commit stays in
its local checkpoint directory. Process-interruption recovery retains the existing
commit ordering; hardware-power-loss durability is not guaranteed.

The single-case/native solver also accepts `--hdf5-root`. When omitted there,
legacy co-located output remains the default, or an existing per-run data link
is used. Existing validation configurations therefore retain their default paths.

## Compiled, parallel convertVTI application

Every normal project build (`make`, `run.sh`, `runContinue.sh`,
`scripts/run_rbc.py`, and therefore `runRBCProduction.sh`) builds
**`build/convertVTI`** alongside the specialized CUDA executable. It is a
standalone C++20 application linked to the HDF5 C library, with no CUDA or Python
runtime dependency. Build only the converter with `make converter` or
`python3 scripts/build_convert_vti.py`. `CXX`, `CXXFLAGS` and `HDF5_FLAGS` support
custom toolchains; otherwise HDF5 flags come from pkg-config.

The public CLI is:

```bash
convertVTI <path/to/hdfs/location> --jobs 4
```

Use `./convertVTI` (the repository launcher for `build/convertVTI`) before
installation, or install into a directory on PATH:

```bash
make install-converter PREFIX="$HOME/.local"
convertVTI /mnt/rbc_hdf5/rayleigh_benard/rbc3_n512_rr3 --jobs 4
```

The input can be one HDF5 file, a fields directory, one HDF5 run directory, the
local run directory with its data link, or an HDF5 root containing multiple runs.
The application discovers volumes recursively and skips metadata, reduced
statistics, validation moments and 2D slices. It infers ZYX dimensions, FP32/FP64
precision and the original run ID from each volume. It requires physical-velocity
metadata and never applies another force shift.

Conversion is **parallel across volumes using CPU worker processes**. Default:
up to four workers, bounded by available logical CPUs; `--jobs N` selects 1..256
workers, with no more processes than input files. Separate processes avoid
requiring a thread-safe HDF5 build. Each worker stages only O(N²) plane buffers,
then writes a scalar temperature and an interleaved three-component velocity
vector in uncompressed VTI. Array order and precision are preserved. No global
3D arrays or populations are introduced. More workers are not a guaranteed
speedup when the drive is saturated. A CUDA backend is not implemented: HDF5
reading/decompression and filesystem writes remain CPU operations, while a GPU
packing kernel would require additional host/device transfers. No performance
claim is made without measurements.

Outputs are `results/<original_run_id>/vtis/`, relative to the converter working
directory; `--results-root DIRECTORY` changes that root. `--step N` filters a
snapshot timestep and may be repeated. Existing outputs are refused by default;
`--skip-existing` resumes conversion after checking expected file size. Atomic
renames, a batch lock and per-output locks preserve completed files on failure.
A worker failure stops the batch; completed files remain. Skipping existing data
checks file size, not a full content checksum. Temporary `.tmp` files from a
terminated worker are replaced on retry; output locks are small persistent files.
The old `scripts/export_rbc_vti.py` command delegates to the compiled executable.

The converter follows the [VTK XML ImageData format](https://docs.vtk.org/en/v9.6.1/vtk_file_formats/vtkxml_file_format.html),
with little-endian appended data and UInt64 byte-count headers. Run it after the
solver has exited; SWMR access to live statistics files is not enabled.

## Storage: decimal TB, no compression savings

Every full timestamp contains **four FP32 scalar fields**:

```
512^3 * 4 fields * 4 bytes = 2 GiB = 2.147483648 GB
```

No compression ratio is assumed. Counts include initial/final snapshots.
The following is field payload only; container/checksum/filesystem overhead,
checkpoints, plots and encoded video are separate. VTI is not counted unless
explicitly produced by the converter.

| Case | Interval | HDF5 volumes | HDF5 GiB | HDF5 decimal TB |
| --- | ---: | ---: | ---: | ---: |
| RBC-1 | 5000 | 512 | 1024 | 1.099512 |
| RBC-2 | 5000 | 615 | 1230 | 1.320702 |
| RBC-3 | 10000 | 512 | 1024 | 1.099512 |
| RBC-4 | 10000 | 512 | 1024 | 1.099512 |
| Total | | 2151 | 4302 | **4.619237** |

Two retained checkpoints per case add about **56 GiB = 0.06013 TB** across the
campaign; raw moments are about 7 GiB per checkpoint. Reduced statistics,
checkpoint headers and retained history copies bring the numeric total to about
**4.680 TB**, before container/filesystem overhead, plots and video. There are
26, 31, 52 and 52 checkpoint writes, but only two pairs per case are retained.
An atomic checkpoint commit temporarily retains a third checkpoint/history pair.
Each explicitly converted VTI adds the same four-field numeric payload as its
source volume (2 GiB); converting RBC-3 alone adds about **1.099512 TB** before
small XML/block/lock overhead. These are stored bytes, not cumulative write traffic.

### Comparison with the previous 10000-step RBC-1/2 output

All four cases have the same 512³ volume size. RBC-1/2 have lower Ra and shorter
run durations; that does not make an individual snapshot smaller. A 5000-step
interval offers more offline temporal samples but does not improve the existing
online diagnostics, which remain sampled every 100 steps.

| Case | Volumes at 10000 | HDF5 TB at 10000 | Volumes at 5000 | HDF5 TB at 5000 |
| --- | ---: | ---: | ---: | ---: |
| RBC-1 | 257 | 0.551903 | 512 | 1.099512 |
| RBC-2 | 308 | 0.661425 | 615 | 1.320702 |

With RBC-3/4 at 10000, the configured campaign field payload is **4.619237 TB**,
an increase of **1.206886 TB** over the previous all-10000 schedule. With two
retained checkpoints per case and reduced statistics, estimate **4.680 TB**
instead of **3.473 TB**, still excluding VTI and container overhead. The configured
defaults are 5000 for RBC-1/2 and 10000 for RBC-3/4; no override is needed.

## HDF5 schema and numerical diagnostics

- `metadata.h5:/metadata` attributes preserve existing names (`NX`, `RA`, `NU`,
  `H`, relaxation, wall/moment/velocity conventions and schedules), plus actual
  HDF5 directory and `vti_enabled=false`. Values are strings parsed by the readers.
- `statistics.h5:/statistics/timeseries`: extendible `(samples,21)` FP64 table.
  The `columns` attribute preserves the five independent Nusselt estimates,
  means/RMS, kinetic energy, both dissipations, extrema, flux uniformity and mass.
- `/statistics/profiles`: `(NZ,19)` FP64 cumulative mean/RMS, convective/conductive
  fluxes, Nu(z), dissipation and Kolmogorov profiles; empty before averaging.
- `/statistics/convergence`: `(5,5)` FP64 Welford summary. Estimator order is
  bottom/top/volume/kinetic/thermal. `naive_stderr` is uncorrected for correlation;
  the Python reports derive correlation-aware approximate confidence intervals.
- `/statistics/runtime`: extendible FP64 timing table. Its `columns` attribute
  records session start, percentage, elapsed/window seconds, sustained window/
  cumulative MLUPS, computation/output seconds and ETA.
- `fields/step_*.h5:/fields/{temperature,ux,uy,uz}` has ZYX shape `(NZ,NY,NX)`.
  Velocities are physical u. File attributes record step, run ID, axis order,
  lattice spacing and velocity convention. Slices have ZX shape and a Y index.
- Lossless shuffle + gzip level 1 (configurable 0..9; zero disables gzip/shuffle)
  and Fletcher32 checksums are used. Chunks: 3D `(1,min(64,NY),min(128,NX))`,
  slices `(min(64,NZ),min(128,NX))`, tables `(128,columns)`. No lossy scale-offset.
- `results/<run_id>/diagnostics.h5` contains derived reduced tables and uncertainty.
  Numerical CSV/VTI output is absent from the solver. Binary `.chk` files retain
  the existing internal checkpoint format, distinct from scientific exports.

The existing Nusselt/dissipation definitions and geometry caveats are documented
in [the numerical RBC guide](rbc.md). Reports retain Nu(z), layer resolution,
local Kolmogorov diagnostics, correlation/batch errors and stationarity cautions.
`dns_convergence_established` stays false; one production grid is not a grid study.

## Runtime and restart semantics

Every save/sampling point and heartbeat prints a flushed line with step, global
completion percentage, session elapsed wall seconds, window/cumulative sustained
MLUPS and ETA. Initial output reports unavailable rates until the first window.
One update means **one complete coupled hydro/thermal lattice cell step**, not
two independent population updates. Numerators count all lattice nodes, including
the plates. Timing uses a monotonic wall clock with CUDA synchronization at
observation points. Sustained MLUPS includes both solvers, boundaries, GPU
statistics, host statistical accumulation, output and diagnostic overhead.

`compute_s` includes the coupled numerical work and statistics, excluding measured
HDF5 writes, field extraction/staging, checkpoint transfers/writes and associated
output work. `output_s` measures those output regions separately. `compute_MLUPS`
uses the remaining time. Runtime-record serialization is charged to the next
window (the final elapsed line includes it). Initial allocations, initialization,
initial snapshots, compilation and restore are outside the timing session. ETA
uses session cumulative end-to-end throughput and is an estimate, especially
before representative checkpoint/3D writes have occurred. On resume, timing
counters restart and `session_start` identifies the segment; scientific
accumulators retain their exact checkpoint state.

Binary checkpoint headers, moment order, precision checks, Welford/Kahan state
and restore validation remain unchanged. A closed `statistics.h5` history in the selected HDF5 location is
copied to a temporary sidecar and renamed before the corresponding `.chk`
commit. On restore that history replaces the active HDF5 file, then rows and
snapshots beyond the committed step are rolled back/replayed and profiles are
recomputed from checkpoint accumulators. This also recovers an interrupted HDF5
append. The latest two checkpoint/history pairs are retained. As before, atomic
renames protect process interruption on a local filesystem; this is not an fsync
or hardware-power-loss durability guarantee.

Legacy checkpoints can be used with explicitly migrated reduced diagnostics:
stop the old solver first, then run
`python3 scripts/convert_rbc_hdf5.py PATH_TO_LEGACY_RUN`. It never deletes legacy
files or replaces an existing HDF5 file. Rebuild/resume with the original case,
grid, precision, RR and sample/transient schedule. The production driver refuses
to adopt unowned legacy directories; use the single-case launcher for those.
Legacy validation snapshots stay readable; newly generated ones use HDF5.

## Commands

The optional [two-GPU PCIe backend](multi-gpu.md) partitions Z only and preserves
all scientific schedules, run IDs and global checkpoints. Add `--gpus 2
--devices 0,1 --halo-transport peer` to the production commands below. Peer mode
requires bidirectional CUDA P2P; `--halo-transport host` explicitly selects pinned
staging. The default remains one GPU. `--split-z K` compiles an optional unequal
partition; execution settings may change on resume without changing physics.

```bash
# Inspect paths/counts only; no build or solver execution.
./runRBCProduction.sh --hdf5-root /mnt/rbc_hdf5 --plan

# Manual full campaign: builds solver + converter; conversion itself stays offline.
./runRBCProduction.sh --hdf5-root /mnt/rbc_hdf5
./runRBCProduction.sh --hdf5-root /mnt/rbc_hdf5 --resume
./runRBCProduction.sh --hdf5-root /mnt/rbc_hdf5 --post-only --no-video

# Parallel offline conversion, installed CLI or ./build/convertVTI.
./build/convertVTI /mnt/rbc_hdf5/rayleigh_benard/rbc3_n512_rr3 --jobs 4
./build/convertVTI /mnt/rbc_hdf5 --jobs 8 --skip-existing --results-root results

# Read reduced diagnostics through the local run's data link.
python3 postRayleigh.py all output/rayleigh_benard/rbc1_n512_rr3

# Inspect RBC-1/2 with their default 5000-step field cadence; no build/run.
./runRBCProduction.sh --only 1,2 --hdf5-root /mnt/rbc_hdf5 --plan
```

`--only`/`--skip` select cases; `--sample`, `--field-every`, `--slice-every`,
`--profile-every`, `--progress-every`, `--checkpoint-every`, `--output-start`,
`--keep-checkpoints`, `--compression`, `--slice-velocity` stay configurable.
`--hdf5-root`, `--output-root`, `--results-root` select the three storage locations.
`--no-video` disables the RBC-3 preview. A fresh invocation skips completed jobs,
while interrupted output requires explicit `--resume`. A matching completed post
receipt skips analysis; `--post-only` regenerates it. Failures retain prior results.
Repeat the original configuration/location overrides on resume or post-only.

## Changed files and verification limits

New compiled tool/build files: `tools/convertVTI.cpp`,
`scripts/build_convert_vti.py`, `Makefile`, and the root `convertVTI` launcher. All existing launchers build the
converter alongside the solver. `scripts/export_rbc_vti.py` is a compatibility
entry point only. The HDF5 relocation touches `src/output.cuh`, `src/main.cu`,
`src/checkpoint.cuh`, `src/validation/output.cuh`, the production/single-case
launchers, `scripts/rbc_hdf5.py`, `postRayleigh.py`, `postCommon.py`, and the
validation reader. No collision, forcing, NEE or statistical reduction is changed.

Subsequent focused CUDA checks cover the single-GPU numerical suite, relocated
HDF5 fields/statistics and cross-mode checkpoint restart; see the
[performance report](performance-rbc.md) and [two-GPU verification limits](multi-gpu.md).
The production campaign, full-volume conversion workload and video rendering
have not been executed. Physical two-GPU PCIe execution and scaling require
verification on the intended pair; only one physical GPU was available here.
