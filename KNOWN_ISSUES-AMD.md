# Known issues — Warp development branch on ROCm / gfx1151

Scope: the HIP/ROCm port of Warp's current main on gfx1151 (Radeon 8060S, Strix
Halo, RDNA 3.5, wave32). Each entry gives the symptom, what it affects, the
workaround if one exists, and its status.

- Status **open**: a defect not yet fixed in the port.
- Status **platform limitation**: a property of the hardware, driver, ROCm
  runtime or a third-party package; not fixable in Warp.

## 1. Behaviour differences on HIP (by design)

### Conditional graph nodes are emulated on HIP

HIP has no conditional graph nodes. Inside a capture, `wp.capture_while()`
and `wp.capture_if()` split it: the running capture ends as one graph, each
body is captured as its own graph, and capture continues into the same
`wp.Graph`. `wp.capture_launch()` replays the pieces and reads each condition
back to the host. `wp.is_conditional_graph_supported()` returns `True`.

Differences from CUDA:

- Every stream forked from the capturing stream must be joined back into it
  before a conditional is recorded. Otherwise the call raises a
  `RuntimeError` stating that forked streams "must be joined back into it
  before capture_while()/capture_if() is called", and the streams are left
  capturing.
- A graph containing conditionals that was saved with `wp.capture_save()`
  cannot be loaded and launched on HIP; loading rebuilds it with native
  conditional nodes.
- Launching a split graph blocks the host at each condition read (one small
  stream synchronization per loop iteration or branch). Condition reads from
  concurrent host threads are serialized through one pinned buffer.
- `wp.capture_debug_dot_print()` of a split graph writes one `digraph` per
  piece, in launch order, each preceded by a `//` comment naming the piece.
  The host-side condition reads are not graph nodes and do not appear.

### External events in replayed graphs

On ROCm, an external event-record node in a replayed graph updates the event
only when the node executes; the event is not marked pending at graph launch
as on CUDA. Warp covers the host side: an event recorded with the external
flag during capture falls back to a device synchronize when the host
synchronizes on it. This is a stronger wait than CUDA's, but correct.

The device side is not covered: a captured wait on another graph's replayed
external record can proceed before that record runs. See section 3.

### cuBQL BVH constructor unavailable

cuBQL is CUDA-only and the HIP build defines `WP_DISABLE_CUBQL`.
`wp.is_cubql_available()` returns `False`; requesting the `"cubql"`
constructor for `wp.Mesh` or `wp.Bvh` fails with
`Warp error: cuBQL support disabled (WP_DISABLE_CUBQL)`. Use the `"sah"`,
`"median"` or `"lbvh"` constructors. MuJoCo-Warp's ray and render paths
request cuBQL and fail with this error.

### OpenGL interop always copies

`cuGraphicsGLRegisterBuffer` / `...Image` are not wired to their HIP
equivalents, so `wp.RegisteredGLBuffer` never obtains a graphics resource and
always takes its `fallback_to_copy` path. `cuStreamGetCtx` has no HIP
equivalent and returns `CUDA_ERROR_NOT_SUPPORTED`.

### Graph memory-allocation nodes

HIP capture-time asynchronous allocation does not surface a `MemAllocNode`;
`wp_cuda_graph_insert_alloc_node` returns null. Graph-free nodes are off by
default (see "Illegal memory access during graph replay" in section 2).

### Single-address atomic contention

HIP lowers float32 `atomicAdd()` to a `global_atomic_cmpswap_b32` retry loop
instead of the hardware `global_atomic_add_f32`, because it cannot prove the
target is coarse-grained memory. Under contention every update retries, so
kernels in which many threads add into one address serialize heavily.

On by default: every float32 `atomic_add` (including `atomic_sub` and the
vector, matrix, quaternion and transform overloads) uses the hardware
instruction. `atomic_max`, `atomic_min`, float64 and float16 atomics are
unaffected. Opt out with `warp.config.hip_fast_fp_atomics = False`,
`WARP_HIP_FAST_FP_ATOMICS=0`, or the `"hip_fast_fp_atomics"` module option.

Opt out when a kernel accumulates into host-coherent memory: the hardware
instruction **silently discards** those updates, returning zero with no
error. It is safe on Warp's own allocations, including the UMA hybrid
allocator and `ReadbackBuffer`, which are created coarse-grained. It is not
safe on managed memory Warp did not allocate that way, such as
`CudaManagedAllocator` arrays.
Integer atomics are unaffected.

### Multiply-add contraction differs between kernels

With `fuse_fp` enabled (the default) HIP kernels are compiled with
`-ffp-contract=fast`, which lets the AMDGPU backend decide per kernel which
multiplies fuse into a following add or subtract. Two kernels that perform
the same arithmetic in the same order can therefore differ by a few ULP.
Seen on gfx950 in `test_fsai`: in the FSAI refit kernel, whose loops have
constant bounds, the compiler emits packed fp32 multiplies (`v_pk_mul_f32`)
and leaves many subtractions unfused (138 `v_sub_f32` against 1 in the
construction kernel, which fuses the same expressions into `v_fma_f32`). The
two factors differ in 1 of 324 and 2 of 729 entries by at most 3.1 fp32 eps;
fp64 and fp16 are bitwise equal there. Results are deterministic, and equal
bit for bit with `fuse_fp=False`, at `-O1`, in debug mode and on the CPU
device. Code that needs two kernels to agree bitwise must set the
`"fuse_fp"` module option to `False`.

### 64 KB shared memory (LDS)

gfx1151 LDS is a hard 64 KB; CUDA devices from sm_75 can opt into more. Tile
kernels whose tiles plus framework overhead exceed 64 KB cannot run. Such a
launch is rejected with a diagnostic naming the shortfall, as on CUDA.

### Lockstep wavefronts

RDNA executes a wavefront in lockstep without independent thread scheduling.
Spinlocks built from `wp.atomic_cas`, where one lane must release a lock while
its wave-mates spin on it, deadlock. `wp.atomic_cas` itself is correct.

### Out-of-bounds accesses fault instead of being masked

An out-of-bounds device access that goes unnoticed on CUDA raises
`CUDA error 700: an illegal memory access` on HIP and poisons the context.
Known trigger: batched MuJoCo-Warp under graph capture with the auto-sized
`njmax`/`nconmax`, where a world exceeding the constraint limit writes out of
bounds. Workaround: pass larger limits (for example `--njmax 4096
--nconmax 16384`). The underlying MuJoCo-Warp defect (a partially dropped
contact leaves `efc_address == -1`, which the solver then indexes) belongs
upstream in MuJoCo-Warp.

### Other

- Profiler start/stop is unsupported: `hipProfilerStart` is deprecated and
  returns `hipErrorNotSupported`. Use roctracer or rocTX.
- PTX output, `entry_point_abi='external_constant_params'`, IPC and the
  CUDA arch-suffix feature are CUDA-only.
- HIP's context API cannot represent "no current context".

## 2. Open bugs

### Illegal memory access during graph replay

- **Symptom:** intermittent `RuntimeError` from `wp.capture_launch` with
  `CUDA error 700: an illegal memory access`, followed at exit by error 700
  from `wp_cuda_graph_destroy` and error 3 from
  `wp_cuda_context_pop_current` (the sticky-error tail, not the defect).
- **Affects:** graphs that allocate inside the captured region and are
  replayed many times; seen in Newton examples (`selection_*`, `sensor_imu`,
  `basic_plotting`, `basic_multi_solver_overlay`, `recording`, `robot_h1`,
  `sensor_contact`, `cable_pile`) and MuJoCo-Warp `forward_test`. The
  faulting kernel is MuJoCo-Warp's `primitive_narrowphase`. A single passing
  run is not evidence of stability.
- **Cause:** capture-time allocations get graph allocation nodes but no
  matching free nodes, so every replay re-executes allocations that are never
  released.
- **Workaround:** run eagerly (no capture). Or set
  `WARP_HIP_GRAPH_FREE_NODES=1`, which adds free nodes and instantiates
  without `AutoFreeOnLaunch`; this removes the fault class but breaks graphs
  whose capture allocations outlive the capture (relaunch fails with
  `invalid argument`). Keeping `AutoFreeOnLaunch` together with free nodes
  faults inside `hipGraphLaunch` on ROCm.
- **Status:** open.

### Intermittent wrong BVH query result

- **Symptom:** a device AABB query misses an intersection that the host finds
  (`host_intersected 1 != device 0`), with no memory fault.
- **Affects:** `geometry/test_bvh` `test_bvh_aabb`; the aabb, capsule,
  sphere and ray query tests can also flip under the pooled test runner.
- **Cause:** writes to freshly sub-allocated small blocks are lost inside the
  ROCm runtime's fragment sub-allocator; a BVH descriptor or
  `primitive_indices` reads back as zeros. Present on ROCm 7.2.x and 7.14.
- **Workaround:** none recommended. `HSA_DISABLE_FRAGMENT_ALLOCATOR=1` removes
  the lost writes but makes the first `wp.utils.radix_sort_pairs` call in a
  process fail with `hipErrorInvalidValue` from hipCUB `SortPairs`; Warp logs
  that error and continues, so the first LBVH build in the process is silently
  corrupt.
- **Status:** open (ROCm runtime).

### Device-to-host readback inside a capture is not refused

- **Symptom:** on unified-memory devices, reading a device array back to the
  host (for example `.numpy()`) while a stream is capturing does not fail as
  on CUDA (error 906); it executes, corrupts the capture state, and
  `hipStreamEndCapture` then crashes.
- **Affects:** code that captures a step containing a host readback, such as
  `SolverImplicitMPM`'s host-side convergence loop.
- **Workaround:** do not read back to the host inside a capture.
- **Status:** open (Warp-side guard needed on the UMA readback path).

### torch external capture with a warm kernel cache

- **Symptom:** `Warp error: stream is not capturing` from the
  `interop/test_torch` `test_torch_graph_*` tests when every module loads from
  a warm kernel cache; they pass from a cold cache.
- **Cause:** a race between torch's stream-capture start and Warp's
  `capture_begin(external=True)` check on HIP with torch ROCm nightly wheels.
- **Status:** open.

### Deterministic mode is not bit-reproducible

`warp.DeterministicMode.RUN_TO_RUN` / `GPU_TO_GPU` results differ at ULP
level between runs: the binned-accumulator reduction in `deterministic.cu` is
not bit-reproducible on gfx1151. Status: open.

### Tile block-level axis reduction (Tier 3)

Tile axis reductions over a dimension larger than 256 (the block-level path)
return wrong results. Tier 1/2 and non-axis reductions are correct. Status:
open.

### Tile-path array reduction is slow

On gfx1151 the tile implementation of a whole-array sum is markedly slower
than the SIMT implementation of the same computation. Not caused by wavefront
width, shuffle lowering or LDS capacity. Status: open (performance).

### Newton `cloth_franka` non-finite state

Intermittent `numpy.linalg.LinAlgError: SVD did not converge`: the Jacobian
read back from the device contains non-finite values. Status: open, cause not
identified.

## 3. Platform, driver and ROCm-version limitations

### External-event waits in replayed graphs

The device-side counterpart of the section 1 entry. Causes
`cuda/test_streams` `test_event_elapsed_time_graph` and
`test_event_external` to fail. Platform limitation (ROCm), no workaround.

### Small allocations fail on ROCm 7.14 under capture-heavy load

On ROCm 7.14, `hipMalloc` returns NULL for requests of a few dozen bytes
outside any capture while the device has almost all memory free. Seen in
Newton workloads that capture graphs continuously (`test_mesh_backface`,
partly `test_heightfield` and `test_remesh`); the failing call is an ordinary
`wp.array` in `builder.finalize()`. ROCm 7.2.x does not show it. Mechanism not
established. Workaround: ROCm 7.2.x for affected workloads.

### gfx1036 (Raphael iGPU): unsynchronized work is dropped, hipCUB radix sort is wrong

Seen on the Ryzen 7000 desktop iGPU (gfx1036, RDNA 2) with ROCm from the
TheRock 10.1.0 wheels (HIP 7.16). Both reproduce without Warp.

- **Asynchronous work is silently dropped.** A copy, memset or kernel launch
  that is not followed by a synchronization, issued after the device has been
  idle for about 1 ms, never executes and reports no error; later reads see
  the old contents. Reproducible with plain HIP calls. Workaround: set
  `AMD_SERIALIZE_COPY=3 AMD_SERIALIZE_KERNEL=3` in the environment, which
  makes the runtime wait before and after each copy and kernel.
- **hipCUB radix sort returns wrong results.**
  `hipcub::DeviceRadixSort::SortPairs` returns output that is not sorted; a
  standalone hipCUB program shows it. No workaround. Everything built on
  `wp.utils.radix_sort_pairs` and `wp.utils.segmented_sort_pairs` is
  unreliable there, including LBVH construction, hash grids and sparse-matrix
  assembly, and so are the tests that cover them.

Platform limitation (ROCm runtime on gfx1036). Neither is reported on
gfx1151, and the same hipCUB sorts are correct on gfx950.

### Scratch allocation limit

A kernel whose scratch need exceeds ROCm's `HSA_SCRATCH_SINGLE_LIMIT` gets a new
scratch allocation on every dispatch, which delays each of its dispatches. Large
kernels that spill registers on gfx1151 cross the default limit. When the
variable is unset, Warp sets it to 1 GiB in the process environment before it
initializes HIP; child processes that inherit that environment see the value. If
PyTorch initialized ROCm first, the runtime has already read the limit and Warp
logs a warning; set `HSA_SCRATCH_SINGLE_LIMIT=1073741824` in the environment
before starting Python.

### Look-back kernels hang on freshly mapped pool memory

rocPRIM's decoupled look-back kernels behind `cub::DeviceScan` and
`cub::DeviceRunLengthEncode` (`warp.utils.array_scan`,
`warp.utils.runlength_encode`) can spin forever when their scratch buffer is
memory the stream-ordered pool has just mapped: every workgroup polls the flag
its predecessor stored and reads it as empty, and the next synchronous call
(`hipFree`, a readback) then waits forever with the GPU at full load. Seen on
kernel 7.0.0-42 (Ubuntu 24.04, ROCm 7.14) in Newton's MPM examples, whose first
step frees temporaries between the sort, the encode and the scan. With the
pipeline's runtime binding (no torch, system ROCm) a `mpm_granular` start hung
in 14 of 38 attempts over three batches (6 of 10, 7 of 20, 1 of 8) and in 4 of
12 in an independent interleaved repeat; the rate varied between batches under
conditions that neither the frame rate nor the GPU load distinguished, and the
cause of the variation is not established. Timeouts with the same log
signature occurred three times before, all under kernel 7.0.0-31 and
undiagnosed then: `mpm_multi_material` and `mpm_viscous` on 2026-09-09,
`mpm_granular` on 2026-09-29. The 40 rows on kernel 7.0.0-34 (2026-09-30 to
2026-11-06, 31 of them at night) did not show it.

With the pool's default release threshold of zero, every synchronization
returned all freed pool memory to the driver, so each of those scratch buffers
was freshly mapped. Two measures, each of which removed the hang on its own
(pool default alone 0 of 12 starts, plain scratch alone 0 of 20, both 0 of 28
plus 0 of 12 in the independent repeat, interleaved with the control): the
scratch of `array_scan` and
`runlength_encode` comes from the plain allocator outside graph capture, as
upstream's newer scan code does, and Warp on HIP sets the pool's release
threshold to the maximum at initialization, unless
`warp.config.mempool_release_threshold`, `WARP_MEMPOOL_RELEASE_THRESHOLD` or
another library set one first, so the pool keeps freed blocks across
synchronizations and serves small buffers from memory mapped earlier. The
runtime still releases completed free blocks under rules of its own, and
memory the pool maps for the first time is fresh, so the second
measure reduces fresh mappings rather than removing them. The cause in the
driver is not established.

### Graph-capture memory leak on ROCm 7.2.x

ROCm 7.2.x does not reclaim memory allocated inside a captured graph; every
replay leaks and sustained simulation slows progressively. Reproducible
without Warp. ROCm 7.14 reclaims correctly. Use ROCm 7.14 for sustained
simulation and RL rollouts.

### ROCm 7.14 installation

- Libraries live under `/opt/rocm/core-7.14/lib` behind an
  `/opt/rocm/lib -> /etc/alternatives/rocm-lib` symlink chain with no
  `ld.so.conf.d` entry; `warp.so` does not load until that directory is on
  `LD_LIBRARY_PATH`.
- Packages come from `repo.amd.com/rocm/packages-multi-arch/` (named
  `amdrocm7.14`), not `repo.radeon.com/rocm/apt/`; 7.2.x must be uninstalled
  first.

### HIPRTC compiler crash on uint8 matrix `ddot`

ROCm 7.2.1's clang crashes in `SelectionDAG::FoldConstantArithmetic` when
compiling a uint8 matrix double-dot kernel with `-DNDEBUG` at `-O2`/`-O3`.
HIPRTC compiles in-process, so the crash ends the process. Workaround in the
port: modules larger than `WARP_HIP_HIPRTC_MAX_SRC_BYTES` (default 2 MB; `0`
means always) are compiled out of process with `hipcc --genco`, retrying
without `-DNDEBUG` if the compiler crashes. A small module containing such a
kernel still compiles through HIPRTC; set `WARP_HIP_HIPRTC_MAX_SRC_BYTES=0`.
Platform limitation (ROCm compiler).

### Textures on Instinct (CDNA) devices

CDNA devices (gfx942, gfx950) report no image support
(`hipDeviceAttributeImageSupport` is 0) and every array or texture-object
creation returns "operation not supported" (error 801).
`Device.is_texture_supported` is `False` there, creating a `wp.Texture1D`,
`wp.Texture2D` or `wp.Texture3D` raises a `RuntimeError` naming the device,
and `cuda/test_texture` and `cuda/test_texture_compiler` skip. Platform
limitation.

### Mipmapped arrays

`hipMipmappedArrayCreate` returns "operation not supported" (error 801) on
gfx1151. Mipmapped textures are unavailable; the tests skip.

### Maximum grid size

A 2^33-thread launch exceeds the maximum grid size the driver accepts on
gfx1151 and is rejected with "invalid argument".

### Lost GPU completion signal (host state)

A process can spin indefinitely in a HIP synchronization path (synchronous
free, device-to-host readback, `capture_launch`) at full CPU with the GPU
idle. Observed with a degraded driver state in which GPU interrupt delivery
is broken from boot. Check with
`journalctl -k -b 0 | grep -c "Fence fallback timer expired"`: a nonzero,
rising count identifies that state; reboot. `HSA_ENABLE_INTERRUPT=0` masks it.
Affects `test_optim` `example_fluid_checkpoint` intermittently.

The same spin also occurs sporadically without that driver state. On an
MI355X SR-IOV virtual function (gfx950, ROCm 7.15, amdgpu-dkms 7.1.3, fence
count zero) `test_grad` `test_mesh_grad_cuda_0` hangs in about 1 % of fresh
processes: 6 of 318 runs of the test, 12 of 1320 runs of its body, none in
1080 runs with `HSA_ENABLE_INTERRUPT=0`. The whole module otherwise takes
28 s with a cold kernel cache and 1.2 s warm. The main thread spins in
`sched_yield` inside `hipFree`, called from the scan in
`wp_mesh_create_device` to release its temporary buffer while the scan
kernels are still queued; the GPU is idle and the kernel log is silent.
In one trial, calling `hipStreamSynchronize` or `hipDeviceSynchronize`
immediately before the `hipFree` gave no hang in about 2800 runs, against 3
in about 900 unmodified runs over the same minutes; the trial was cut short
and the change is not adopted. The four hangs whose onset was timed fell
within 21 s of the extrapolated 150 s schedule on which the node's
`amd-metrics-exporter` runs `rocpctl`, which toggles the performance counters
of every compute queue; that link is not established. Reproduce with:

```sh
for i in $(seq 300); do
  WARP_CACHE_PATH=$(mktemp -d) timeout -s KILL 60 \
    python -m unittest warp.tests.test_grad -k test_mesh_grad_cuda \
    >/dev/null 2>&1 || echo "run $i hung or failed"
done
```

Run only a few such loops at a time on a node that runs the exporter; see
"KFD deadlock with the metrics exporter's profiler" below. Platform
limitation; workaround `HSA_ENABLE_INTERRUPT=0`.

### KFD deadlock with the metrics exporter's profiler

amdgpu-dkms 7.1.3 takes two KFD locks in opposite orders. Tearing down a
process that enabled performance counters holds `profiler_lock` and takes the
queue-manager lock (`kfd_process_destroy_pdds` → `set_perfcount`); creating a
compute queue holds the queue-manager lock and takes `profiler_lock`
(`create_queue_cpsch` → `init_mqd`). When a `rocpctl` run of
`amd-metrics-exporter` exits while another process creates a queue, both
block for good, and with them every later open of `/dev/kfd`, queue ioctl and
GPU process exit on the node. Hit once on the MI355X node with 24 concurrent
loops of short HIP processes. Signs: load average climbing with idle CPUs,
unkillable processes in state `D`, and `task ... blocked for more than 122
seconds` in the kernel log with `set_perfcount` and `init_mqd` in the traces.
Only a reboot clears it. Avoid heavy churn of HIP processes on a node whose
exporter profiles the GPUs, or disable the exporter's profiler metrics.
Platform limitation (driver).

### Interop packages

- torch and jax for gfx1151 come from the TheRock nightly index
  (`https://rocm.nightlies.amd.com/v2/gfx1151/`). Paddle has no gfx1151
  build.
- When torch's bundled ROCm and a system ROCm share a process, set
  `LD_LIBRARY_PATH=<site-packages>/_rocm_sdk_core/lib`. Without it, against
  ROCm 7.2: `undefined symbol: hsa_ext_image_create_v2`; against ROCm 7.14:
  `libamd_comgr.so.3: undefined symbol ... LLVM_23.0`.
- Import torch before warp in one process.
- jax 0.11.0 nightly (`jax`, `jaxlib`, `jax-rocm7-*`) aborts inside XLA
  during Warp interop tests; pin 0.10.2.
- MJX/brax PPO training on jax-rocm fails compiling the fused physics graphs
  (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`, or a silent segfault), with
  no Warp in the process. The MJX `impl='warp'` bridge fails on GraphMode API
  skew between mujoco-mjx and released mujoco-warp.

### usd-core heap corruption

`UsdPhysics.LoadUsdPhysicsFromRange` in the usd-core 26.3 wheel corrupts the
glibc heap in its parallel collision-finalize phase (`double free or
corruption (fasttop)`, `malloc(): unaligned tcache chunk`). Not GPU-related;
affects Newton USD imports such as `test_menagerie_usd_mujoco`. Workaround:
set `PXR_WORK_THREAD_LIMIT=1` before any `pxr` work starts. Calling
`Work.SetConcurrencyLimit(1)` around the parse only is not sufficient.

### Pooled test runner

Under `-s autodetect`, test files run concurrently. A genuine fault in one
file aborts the GPU queue and fails unrelated files running at that moment.
Count only failures that reproduce with one test file per process.

## 4. Tests skipped or expected to fail on HIP

Expected to fail:

- `cuda/test_streams` `test_event_elapsed_time_graph`,
  `test_event_external` — external-event waits in replayed graphs (section 3).
- `matrix/test_mat` `test_inverse_float16` — fp16 result differs from the
  expected value by 4 ULP against the test's `atol=0.05`; tolerance deliberately
  not widened.
- `cuda/test_streams` `test_stream_priority_timings` — intermittent; the
  timing assertion reads zero elapsed time for a very short kernel.
- `geometry/test_bvh` `test_bvh_aabb` — intermittent; lost writes (section 2).
- `test_optim` `example_fluid_checkpoint` — intermittent wedge in
  `capture_launch` (section 3).
- `test_grad` `test_mesh_grad` on the GPU — intermittent hang in `hipFree`
  on an MI355X virtual function (lost GPU completion signal, section 3).

Skipped on HIP:

- `test_graph` `test_graph_fill_drives_capture_while` — written for native
  conditional nodes.
- `test_graph` graph memory-node tests — no `MemAllocNode` on HIP.
- `test_apic` loaded-conditional-graph rebuild — needs native conditional
  nodes.
- `deterministic/` GPU tests — deterministic mode not bit-reproducible.
- `tile/test_tile_shared_memory` `test_tile_shared_mem_large`,
  `tile/test_tile_view` `test_tile_assign_2d`, `tile/test_tile_reduce` 3D
  axis-reduce backward — exceed 64 KB LDS.
- `tile/test_tile_reduce` Tier 3 axis reduction — open bug (section 2).
- `tile/test_tile_fft`, `test_tile_solve`, `test_tile_mlp` MathDx cases —
  libmathdx is CUDA-only.
- `cuda/test_texture` mipmapped-array tests — error 801.
- `cuda/test_texture`, `cuda/test_texture_compiler` on devices without image
  support (CDNA) — no textures (section 3).
- `test_large` `test_large_launch_large_kernel` — grid-size limit.
- `test_atomic_cas` spinlock tests — lockstep wavefronts.
- `test_cuda_profiler` start/stop — `hipProfilerStart` unsupported.
- `cuda/test_kernel_attributes` external-constant-params case — CUDA-only.
- `test_fast_math` PTX verification, `aot/test_module_aot` PTX target —
  CUDA-only.
- `cuda/test_ipc`, `cuda/test_cuda_arch_suffix`, `cuda/test_clang_cuda` —
  CUDA-only.
- `test_module_parallel_load` no-current-context postcondition — not
  representable in HIP's context API.
- `cuda/test_async`
  `test_copy_s2s_d2d_SrcPoolOff_DstPoolOff_NoStream_Grad_NoGraph_AccessSrcDst`
  — flaky in serial full-suite runs (HSA signal-pool state); passes in
  isolation.

Downstream (Newton, MuJoCo-Warp):

- Newton `determinism/test_solver_determinism` particle tests — deterministic
  mode (section 2).
- Newton `test_solver_vbd` `test_edge_face_pushes_vertices_out` — the contact
  lands on a face-selection tie of the box SDF; floating-point path
  differences select the other face.
- MuJoCo-Warp `ray_test`, `render_test` and one `io_test` case — cuBQL
  constructor unavailable (section 1).
