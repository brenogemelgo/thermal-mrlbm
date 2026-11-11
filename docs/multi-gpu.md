# Two-GPU RBC execution over PCIe

The RBC executable supports one process using **exactly two CUDA GPUs**, with a
one-dimensional Z decomposition. `--gpus 1` preserves the single-domain path;
`--gpus 2 --devices 0,1` selects two distinct, visible CUDA ordinals. The default
transport is **`cudaMemcpyPeerAsync` in both directions**. No MPI/NCCL installation
or persistent population arrays are required.

## Launch and capability checks

```bash
# Build a small executable and inspect visible devices and the P2P matrix only.
python3 scripts/run_rbc.py --case 1 --grid 16 --test-grid --gpu-info

# Short coupled verification on the actual GPU pair.
python3 scripts/run_rbc.py --case 1 --grid 64 --steps 300 \
  --transient-steps 20 --sample 10 --run-id peer_short \
  --gpus 2 --devices 0,1 --halo-transport peer

# Inspect the unchanged production configuration without compiling/running.
./runRBCProduction.sh --gpus 2 --devices 0,1 --plan

# Launch manually when ready. These commands run the full production campaign.
./runRBCProduction.sh --gpus 2 --devices 0,1 --halo-transport peer \
  --hdf5-root /mnt/rbc_hdf5
./runRBCProduction.sh --gpus 2 --devices 0,1 --halo-transport peer \
  --hdf5-root /mnt/rbc_hdf5 --resume
```

The native executable accepts the same `--gpus`, `--devices`, `--halo-transport`
and `--gpu-info` options. Both kernels are built by the normal launcher. Set
`--arch` to an architecture supported by both GPUs; the existing default is
`sm_86`. `CUDA_VISIBLE_DEVICES` determines the ordinal mapping.

PCIe connectivity alone does not guarantee CUDA P2P capability. Startup checks
both directions and enables peer access. Requested `peer` mode fails explicitly
when the selected pair cannot support it. `--halo-transport host` uses two-stage
asynchronous copies through portable pinned host memory; `auto` explicitly opts
into capability-based fallback. The selected transport and slab ranges are
printed and included in HDF5 metadata. Querying capability does not measure
effective bandwidth or prove that a particular PCIe topology is fast. See the
[CUDA peer-copy API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html)
for the underlying interface.

The default compile-time split is `NZ/2`. For unequal GPUs, `--split-z K` in the
Python/production launchers specializes GPU 0 to `[0,K)` and GPU 1 to `[K,NZ)`.
Each slab must contain at least two planes. Tune K from measured throughput and
available memory; an unequal split is not automatically faster. The binary name
includes an overridden split, while the scientific run ID remains unchanged.
GPU count, device selection, transport and split are execution settings, separate
from the immutable production physics/output configuration.

## Timestep and ownership

The physical coordinates and plate separation remain global: `H=NZ-1`.
Periodic X/Y neighbors are local to each full XY plane. Only `z=0` and `z=NZ-1`
are NEE plates; the internal slab interface is an ordinary fluid plane. Collision,
RR2/RR3 reconstruction and the force-shifted stored velocity use the existing
shared node implementation with global source/destination coordinates.

Each GPU owns two SoA moment buffers and two incoming halo buffers. The interface
kernel reconstructs, streams and collides one owned plane, then writes its 14
moments to both the owned buffer and a contiguous send face. Packing therefore
requires no additional per-step kernel or read of the updated interface. One
peer DMA per direction transfers that face. No GPU reads remote moments in its
collision kernel.

The interior and interface are separately specialized. Direction-dependent
interface pulls select local/halo memory at compile time; bulk pulls have no halo
branch. Both use coalesced XY accesses. The existing fused NEE update remains
owned by the adjacent interior node and needs no inter-block synchronization.
Initialization uses a separate local plate pass before the first halo exchange.

Each GPU has a high-priority interface stream, a high-priority transfer stream
and a normal-priority interior stream. Host staging also uses an upload stream.
The interface is queued first, followed immediately by its copy and the interior,
so communication can overlap interior computation. There are **two computation
kernels per GPU per ordinary timestep**, with no standalone packing/boundary
kernel. Combining the entire slab into one ordinary CUDA kernel would defer the
outgoing copy until the whole slab finished, removing this overlap.

Dependency events have timing disabled. For step t:

- Interface t waits for local interior t-1, incoming face t-1 and outgoing-buffer
  transfer t-2. Its own stream also serializes earlier interface work.
- Interior t waits for interface t-1; its own stream serializes earlier interior
  work. This protects both alternating input/output buffers.
- Face transfer t waits for its producer interface t and destination interface
  t-1, protecting reuse of the alternating incoming halo.

All waits capture an already-enqueued recording before that event slot is
reused. Cross-device waits order the copies without a CPU barrier; completion
events for staged H2D copies belong to the destination device. CUDA documents
[cross-device event waits](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/)
and [captured event-record generations](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html).
Ordinary timesteps perform no host/device-wide synchronization, allocation or
host data transfer in peer mode. Actual overlap depends on GPU resources and the
PCIe topology; this implementation does not guarantee linear scaling.

## Statistics, scientific exports and restart

Statistics run only at the configured sampling interval. Each GPU reduces its
owned planes, reading the incoming one-plane halo for derivatives at the slab
interface. FP64 reduced profiles are copied concurrently into disjoint portions
of one pinned global profile table. Existing host Kahan/Welford accumulation,
all five Nusselt estimates and export schedules then operate on that table.
No full fields are transferred for statistics.

Sampling waits for the necessary interface/halo events, then synchronizes the
two reduction streams before host accumulation. Field/checkpoint output uses
the existing explicit observation-point synchronization. These operations are
outside the asynchronous ordinary-timestep path and included in sustained
aggregate MLUPS, using the **global** lattice update count.

HDF5 schema, global field ordering, original run IDs and data/results locations
are preserved apart from the subsequently requested temperature-only 3D field set.
Temperature capture copies disjoint local SoA ranges directly into one shared
pinned host volume, with no full-domain GPU gather or packing kernel. Both GPU
copies are enqueued before waiting for completion. The separate CPU writer then
compresses/writes while timestepping resumes; see [the export guide](export-pipeline.md).
Checkpoints retain all moments and stage at most 32 global XY
planes of one moment at a time (32 MiB at 512³ FP32). The existing checkpoint
header, 14-field order and statistical accumulators are unchanged. A restart
may change one/two-GPU execution or slab split while retaining the same global
case, grid, precision and numerical/statistical configuration. The driver keeps
its existing completed-run protection and explicit `--resume` requirement.

## Memory and communication at 512³ FP32

For a balanced split:

| Allocation / traffic | Size |
| --- | ---: |
| Two owned 14-field moment buffers, per GPU | 7 GiB |
| One packed XY face | 14 MiB |
| Double-buffered send and receive faces, per GPU | 56 MiB |
| Local FP64 reduction scratch and profile, per GPU | about 36.04 MiB |
| Core allocations, per GPU | about 7.090 GiB |
| Lazy physical field extraction scratch, per GPU | up to 4 MiB |
| Peer traffic, each direction per timestep | 14 MiB |
| Peer traffic, both directions per timestep | 28 MiB |

These figures exclude the CUDA context, allocator rounding and display/other
applications. Startup checks currently available memory before allocating core
buffers. Optional output allocation still needs headroom; available memory on an
8 GiB card depends on its other users. FP64 doubles moments and halo sizes.
Host fallback adds 56 MiB of pinned halo staging across both devices. The global
pinned FP64 profile table is only 72 KiB. Wire time is bounded below by face bytes
divided by measured effective PCIe bandwidth; duplex topology and overlap must
be measured on the intended pair.

## Focused validation and performance limits

```bash
# Short analytical/coupled cases, outputs, cross-mode restarts and sanitizers.
python3 tests/validate_multi_gpu.py --sanitizers
python3 tests/validate_multi_gpu.py --quick --split-z 2 --sanitizers

# Short real-GPU scaling comparison; no emulated performance results.
python3 scripts/benchmark_multi_gpu.py --grid 128 --steps 300 --repeats 5 \
  --devices 0,1 --halo-transport peer

# Representative 512³ two-GPU run when a single-GPU baseline cannot fit.
python3 scripts/benchmark_multi_gpu.py --grid 512 --steps 300 --repeats 3 \
  --devices 0,1 --only-two
```

The correctness script uses test-build-only one-device emulation as well as real
peer/host execution when two physical devices are available. It checks minimal,
odd and aligned grids, RR2/RR3 conduction, manufactured thermal advection and
forced hydrodynamics, and a 257-step interval without intermediate sampling/output
barriers. It compares all moments, fields, slices, diagnostics and restarts from
single to distributed, distributed to single, and between transports. FP32
moments are compared bitwise; FP64 comparisons permit 32 machine epsilons on the
unit-density lattice scale for force-corrected stresses near cancellation.
Compute Sanitizer covers memory, race, synchronization and initialization checks
for both emulated exchange paths. It cannot prove physical cross-device ordering.

The real-GPU benchmark alternates matched single/two-GPU runs, retains checkpoints
and writes `report.json`, `timings.csv`, build resource-usage logs and per-run
logs in a new work directory. Statistics and final checkpoint IO are included;
computation and output timing are also reported separately. `--only-two` omits
the single-device baseline. Unsupported hardware is reported as SKIP or FAIL,
never substituted with an emulated speedup. Use Nsight Systems on that same short
native command to inspect copy/interior overlap and PCIe serialization.

Only one physical RTX 3050 Laptop GPU was available during implementation.
The emulated tests and existing single-GPU regression suite passed. All compared
physical fields and diagnostics matched exactly; the FP64 manufactured channel
had a maximum stress difference of `2.12e-22`. An uneven 2/5-plane split passed
with bitwise-equal FP32 moments, including its restart and sanitizer checks.
The SM86 64³ build used 48/56 registers for the two interior specializations and
48 for the interface kernels, with zero spill stores/loads and no stack frame in
those four kernels. The single-GPU kernel retained 56 registers and zero spills.

A short 64³, 200-step, three-repeat single-device comparison against the saved
pre-change source gave median sustained 356.269 → 360.870 MLUPS, with substantial
desktop clock/timing variation; it does not establish a speedup. Moments and
diagnostics were bitwise equal. **Physical two-GPU P2P execution, cross-device
event behavior, bandwidth and scaling remain unverified on the target pair.**
No production simulations were launched and no DNS convergence is claimed.

The local [machine-readable report](../results/multi_gpu_20261110/report.json)
collects numerical comparisons, sanitizer results, resource logs, the short
single-device comparison and the explicit physical-peer SKIP result.

## Implementation files

- `src/multiGpu/{slab,kernels,runtime}.cuh`: specialized ownership/access views,
  kernels, stream/event scheduling, transports and device-local allocations.
- `src/momentStorage.cuh`, `src/kernel.cuh`, `src/reconstruction.cuh`,
  `src/initialConditions.cuh`, `src/boundary/deviceRuntime.cuh`: shared node
  operations adapted to local storage without changing the numerical formulas.
- `src/rbcStatistics.cuh`: reusable reduction bodies and existing host
  accumulation from an assembled reduced profile.
- `src/hostMomentIO.cuh`, `src/output.cuh`, `src/checkpoint.cuh`,
  `src/validation/output.cuh`: bounded distributed staging with unchanged global
  scientific data/checkpoint layouts.
- `src/main.cu`, `scripts/run_rbc.py`, `scripts/rbc_production.py`: execution
  selection and launch integration.
- `tests/validate_multi_gpu.py`, `scripts/benchmark_multi_gpu.py`: correctness,
  sanitizers and short real-pair timing automation.
- `README.md`, `docs/production-rbc.md`, this guide: usage and limitations.
