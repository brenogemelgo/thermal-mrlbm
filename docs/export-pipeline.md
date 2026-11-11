# Temperature snapshots and the HDF5 export pipeline

New 3D snapshots export **temperature only**. This is the requested change to the
field set; all other scientific contracts remain: FP32/FP64 precision, ZYX shape,
dataset name `/fields/temperature`, file attributes, timestep ordering, initial/
final output, directory paths, compression level and snapshot intervals.
RBC-1/2 retain 5000-step fields; RBC-3/4 retain 10000-step fields. Statistical
sampling stays at 100 steps, and production checkpoints stay at 100000 steps.
No collision/streaming kernel, physical model or statistical accumulator changed.

The five Nusselt estimators, mean/RMS profiles, dissipations, uncertainty and DNS
diagnostics continue to use reduced online statistics. The existing Python video
reads temperature as before. `convertVTI` now accepts temperature-only volumes;
legacy temperature + physical velocity volumes still produce the original vector
VTI. Explicitly requested `--slice-velocity` retains its existing 2D uz output.
Checkpoints retain all fourteen moments and the complete statistical state.

## Execution and resource ownership

```bash
# Manual production launch, existing paths and schedules.
./runRBCProduction.sh --gpus 2 --devices 0,1 --hdf5-root /mnt/rbc_hdf5

# Resume an existing campaign with the same original physics/output options.
./runRBCProduction.sh --gpus 2 --devices 0,1 --hdf5-root /mnt/rbc_hdf5 --resume

# Blocking parallel writer for comparison.
./runRBCProduction.sh --field-io sync --io-workers 4 --hdf5-root /mnt/rbc_hdf5

# Bounded plane fallback when a complete host snapshot cannot be pinned.
./runRBCProduction.sh --field-io serial --hdf5-root /mnt/rbc_hdf5
```

These are manual launch examples; no production runs were performed during
implementation. The options also work in `scripts/run_rbc.py` and the native
solver. `--io-workers` selects 1..32 CPU compression workers; the default is four.
`--compression` keeps its existing meaning (default gzip 1, zero disables
shuffle/gzip while retaining Fletcher32). The default path is `--field-io async`.
Tune workers on the destination system; oversubscribing a busy host can hurt.

Temperature is already a contiguous moment in the solver's SoA layout. The
exporter copies that range directly into **one reusable pinned shared host
buffer**, with no GPU packing kernel, no GPU staging volume, no host repacking
and no pageable staging copy. At 512³ FP32 the buffer is 512 MiB; FP64 requires
1 GiB. Memory is allocated/registered once on first use. There is no unbounded
queue or second full host copy. The `serial` fallback uses bounded plane staging
and standard blocking HDF5 writes instead of pinning a whole volume.

On one GPU, a readiness event fences the default numerical stream. On two GPUs,
each capture waits only for the owned interior/interface events and enqueues a
contiguous copy of its local temperature into its disjoint global host range.
Both transfers are enqueued before waiting for completion. Timing events use
each copy's own device. Capture finishes before returning control to the solver,
so the next alternating-buffer update cannot overwrite data being transferred.
Overlapping D2H with subsequent updates would require additional GPU staging or
solver buffer-reuse dependencies; neither is introduced here.

After capture, the parent sends a small request over a Unix sequence-packet
socket. A separate process, started with `posix_spawn`/exec of the same binary,
reads the shared host buffer and handles CPU encoding and HDF5 writing. Its
worker entry runs before any explicit CUDA API call. This avoids using CUDA
after fork and avoids HDF5's global library lock contending with the parent's
frequent statistics writes. No separate executable installation is needed and
non-thread-safe serial HDF5 builds are supported. Linux `/proc`, memfd and POSIX
process/socket interfaces are required, as used by the current CUDA workstation
workflow.

Only the writer's main thread calls HDF5. Persistent CPU workers encode bounded
batches of independent chunks with reusable raw/shuffle/compressed buffers and
reusable zlib streams. Output uses the existing chunk dimensions
`(1,min(64,NY),min(128,NX))`, shuffle, the configured zlib level and Fletcher32.
`H5Dwrite_chunk` stores these already encoded chunks in normal HDF5 datasets;
readers use the standard built-in filters. Chunk writes bypass repeated HDF5
hyperslab/dataspace creation and the raw chunk cache. Zero-padded edge chunks,
little-endian data and checksum byte ordering are tested against the original
HDF5 encoder. This follows the
[HDF5 direct chunk interface](https://support.hdfgroup.org/documentation/hdf5/latest/group___h5_d.html)
and the standard
[Fletcher32 filter contract](https://github.com/HDFGroup/hdf5/blob/hdf5-1_10_7/src/H5Zfletcher32.c).

The next snapshot applies backpressure if the previous write is still running.
Only an acknowledgement after closing and publishing the previous file releases
the host buffer. Initial output drains before the timed simulation session.
Snapshots through a checkpoint timestep drain before its existing checkpoint/
history transaction. Final output drains before final elapsed/MLUPS reporting.

## Failure and restart handling

Each file is written under its existing `step_*.h5.tmp` name, flushed and checked
on close, then renamed to `step_*.h5`. A complete existing snapshot is refused by
the new writer; it is never silently replaced. Failed writes/worker death fail
the parent at an acknowledgement or submission boundary. Completed files remain;
an incomplete `.tmp` is unpublished and can be replaced when that step is replayed.

The child requests termination if its parent dies. Exceptional cleanup stops and
reaps the child before unpinning/unmapping shared memory. Normal cleanup closes
the command channel and reaps the idle child. Socket sends suppress SIGPIPE and
transport failures become explicit errors. CUDA resources remain owned by their
creating device and are reused across snapshots.

Existing checkpoint/header/history formats are unchanged. As before, checkpoint
state is authoritative and snapshots newer than it are removed/replayed during
resume. No checksum, statistic or numerical precision is dropped to accelerate
scientific output. Atomic rename protects process interruption; HDF5 flushing
does not provide an fsync/power-loss durability guarantee, matching the previous
implementation.

An interrupted **production-driver** run whose only configuration difference is
the old four-field export list may transition to temperature-only with explicit
`--resume`. The original configuration is recorded in
`production_export_transition.json`. Earlier four-field snapshots remain intact;
new steps contain temperature only. Completed legacy runs retain their original
manifest and completion receipt. All other configuration mismatches still fail.
Single-case/native checkpoint resumes also retain their normal numerical checks.

## Timing interpretation

Existing `/statistics/runtime` columns remain unchanged. `output_s` measures
foreground output work: capture overhead, transfer, queue backpressure, checkpoint
IO, profile serialization and waits needed to drain outstanding writes. Waiting
for previously queued numerical work is charged to computation. Nested output
regions are charged once. `compute_s=elapsed_s-output_s` includes the coupled
solver and statistics, as before, and includes any contention from the writer.

The additional console `writer=...s` counter is the summed wall duration of
**completed** background writes in the current timing session. It overlaps computation and must not be added
to elapsed time or subtracted from `compute_s`. End-to-end MLUPS includes all
elapsed work and final drain, so background output cannot create a misleading
completion-time speedup.

An additive `/statistics/exports` table records completed snapshots:

- step, uncompressed payload bytes, GPU packing seconds (zero for temperature);
- GPU D2H seconds, capture wall seconds and queue backpressure;
- encoding wall seconds and summed encoder **thread CPU** seconds;
- direct HDF5 chunk write, flush/close and total writer seconds;
- actual file bytes, submission-to-observed-completion latency, workers and level.

For two GPUs, D2H seconds sum both CUDA-event spans around the copies; capture wall time
includes their overlap. Observed completion latency includes time until the
parent polls, rather than measuring only file publication. Direct write/flush
timers measure HDF5/VFD/OS-buffered operations, not guaranteed physical-media
completion. The old `H5Dwrite` timing necessarily combines filtering and writes;
the isolated encoder explicitly separates those stages in the new path.

## Reproducible short checks and benchmarks

```bash
# Save src/ and cases/ before making changes when preparing a new comparison.
python3 tests/validate_export.py --before /path/to/original-tree
python3 tests/validate_multi_gpu.py --quick --sanitizers

# Identical changing fields and settings; original 4-field and temperature-only
# plane paths, plus 1-/4-worker sync/async writers. No solver evolution.
python3 scripts/benchmark_export.py --grids 128 512 --snapshots 2 --repeats 1

# Explicit experiments only; these do not change production defaults.
python3 scripts/benchmark_export.py --grids 128 --snapshots 2 --repeats 1 \
  --workers 4 --compression 0 1 --chunks 64x128 128x128 128x256

# Actual short coupled throughput under identical test schedules.
python3 scripts/benchmark_rbc.py --before /path/to/original-tree --before-fused \
  --grids 64 --steps 300 --sample 100 --repeats 5 --field-every 50

# Failure injection using the export harness binary produced by the benchmark.
python3 tests/validate_export_failures.py /path/to/benchmark/export-n128
```

The export microbenchmark changes its GPU source between requests, checking that
capture and backpressure prevent mixed snapshots. It stages synthetic fields to
permit a representative 512³ IO test on the available 4 GiB GPU; allocating all
fourteen solver moments at that grid would not fit. It is not a turbulence or
production-throughput benchmark. Back-to-back requests intentionally exercise
queue backpressure. The coupled test uses smaller grids and a short explicit
test cadence, with the same settings before/after.

Validation checks temperature bits, compressed chunk bytes, filters, attributes,
FP64 padded edge chunks, compression levels 0/1/6, all solver moments, online
statistics, unchanged optional slices, old-to-new/new-to-new restart, stale temp
files and legacy/scalar VTI conversion. Artifacts and measured results are kept
in `results/export_20261110/`. Only one physical GPU was available; actual two-GPU
capture remains to be measured/verified on the target pair. No production run or
DNS convergence claim is made.

## Measurements on the available workstation

Hardware: one RTX 3050 Laptop GPU (4 GiB), local NVMe filesystem, HDF5 1.10.7,
zlib, Linux. These measurements use deterministic synthetic data; they do not
promise the same latency for turbulent fields, a different CPU or another drive.

512³ FP32, gzip 1, original `(1,64,128)` chunks, two changing snapshots:

| Path | Payload/snapshot | Packing GPU | D2H | Filtering + HDF5 writes | Total blocking/worker time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original four-field plane exporter | 2 GiB | 0.081 s | 0.505 s | 35.197 s | 35.904 s blocking |
| Original temperature-only plane pipeline | 512 MiB | 0.025 s | 0.138 s | 6.056 s | 6.254 s blocking |
| New four-worker encoder/background writer | 512 MiB | 0 | 0.063 s | 1.969 s encoding + 0.172 s raw writes | 2.152 s writer |

The original four-field pipeline spent about 98% of its measured wall time in
HDF5 filtering/write calls. The new **single-worker** encoder took 5.968 s for
encoding versus 0.176 s for raw HDF5 writes, independently identifying CPU
filtering as the main cost on this workload. Four workers reduced encoding wall
time to 1.969 s. Larger `(1,128,128)` chunks did not improve the 128³ test; the
default geometry is retained. Compression zero was faster, but it changes file
size/settings, so the production default remains gzip 1.

The fully drained two-snapshot asynchronous test took 4.660 s, equivalent to
**219.7 MiB/s of uncompressed temperature payload**, versus 81.9 MiB/s for the
original temperature-only plane path. Back-to-back submissions deliberately
encountered backpressure: 0.063 s is the capture stage, not a promise that a
complete file is published that quickly. Startup pinning/spawn is additional,
and normally occurs with the initial snapshot outside the timed simulation.

A separate coupled 64³ comparison used 300 steps, samples every 100 steps,
fields every 50 steps, a final checkpoint and five alternating matched trials.
The original four-field exporter achieved a median **162.495 MLUPS**; the new
temperature-only asynchronous path achieved **451.966 MLUPS** (**2.78x**).
Median compute MLUPS was 554.909 versus 547.602, within desktop clock/timing
variation; the improvement comes from output. Median foreground IO fell from
0.345 s to 0.030 s. All moments and scientific diagnostics were bitwise equal.
These are short representative timings, not production-grid solver timings.

Temperature values, compressed temperature chunks and metadata other than the
intentional distinct comparison run IDs matched exactly. FP64 padded-edge tests
passed at levels 0, 1 and 6. Old/new restart and scalar/legacy-vector converter
tests passed. GPU memory/race/synchronization/initialization checks reported no
errors, and write-error/worker-death/parent-death fault injection preserved all
completed files. The existing Python diagnostics and temperature-video path were
also exercised on short temperature-only output.

Modified implementation files are `src/output.cuh`, `src/temperatureOutput.cuh`,
`src/snapshotWriter.hpp`, `src/hdf5Output.cuh`, `src/hostMomentIO.cuh`,
`src/multiGpu/runtime.cuh` and `src/main.cu`; launcher/build/storage integration is
in `scripts/run_rbc.py`, `scripts/rbc_production.py`, `scripts/build_convert_vti.py`,
`run.sh` and `runContinue.sh`. `tools/convertVTI.cpp` accepts the new field set.
Benchmark/validation additions are `tests/benchmark_export.cu`,
`tests/validate_export.py`, `tests/validate_export_failures.py`,
`scripts/benchmark_export.py` and optional export controls in
`scripts/benchmark_rbc.py`. README and the production/multi-GPU guides are updated.
The pre-existing `src/kernel.cuh` workspace changes were left untouched and its
SHA-256 stayed identical throughout this task.
