# Warp on AMD HIP/ROCm

AMD HIP/ROCm build of NVIDIA Warp, development branch: the port on NVIDIA's current main.

## Quickstart

Use a clean, **isolated environment** (installing into the system Python often
fails because build/runtime dependencies are missing). Conda below is the
turnkey option; if you already have Python 3.11/3.12 a plain `venv` works too —
see [Environment setup](#environment-setup). The short version:

```bash
# 0. Clone this branch (the repository is public). git-lfs must be installed
#    FIRST — the USD/NVDB test and example assets are LFS objects, and without
#    it the clone silently contains pointer files.
git lfs install
git clone -b amd-integration-halo https://github.com/AMD-Ecosystem/warp
cd warp

# 1. Create and activate an environment (Python 3.12; 3.11 also works)
conda create -n warp-amd python=3.12 -y
conda activate warp-amd

# 2. Install build dependencies
pip install numpy setuptools packaging wheel
# Optional, only for Warp's torch interop — NOT needed for Warp or Newton.
# PyTorch for gfx1151 comes from AMD's gfx1151 index (the stock
# download.pytorch.org ROCm wheels do not carry gfx1151 kernels; rolling
# nightly index):
#   pip install torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/

# 3. Build Warp for HIP. The compile takes about 1-2 minutes at the default
#    8 jobs; the first-ever build additionally downloads the
#    LLVM/Clang toolchain dependency. Must run before step 4:
#    `pip install -e .` expects warp/bin/warp.so to already exist.
#    build_amd.sh auto-detects python/python3; override with
#    PYTHON=/path/to/python
./build_amd.sh

# 4. Install into the active environment, then sanity-check the build.
#    On ROCm 7.14, export LD_LIBRARY_PATH first (see Prerequisites) or
#    warp.so will not load.
pip install -e .
python tools/run_gfx1151_smoke.py
```

Recommended ROCm version: **7.14.1**. See [Choosing a ROCm version](#choosing-a-rocm-version),
including a validated container route that needs no host ROCm upgrade.

A passing smoke test means the build loads and runs kernels. It does **not**
mean the build is sound: it runs two elementwise kernels on 1024 floats and
exercises no graph capture, no sustained execution and no BVH or mesh query, so
it cannot detect any of the defects in `KNOWN_ISSUES-AMD.md`. Sustained
throughput in particular must be measured over minutes, because the ROCm 7.2.x
graph-capture leak only becomes visible after a few hundred frames.

To run Newton physics examples (ANYmal, cloth, softbody, etc.) see
[Newton integration](#newton-integration).
(`torch` in step 2 is optional; it is not required for Warp or Newton.)

## Supported hardware

This port is **validated only on `gfx1151`** (AMD Radeon 8060S / Strix Halo iGPU). Other RDNA architectures are buildable but untested by the port maintainers.

| Architecture | Status | Notes |
|---|---|---|
| `gfx1151` | Validated | Known issues in `KNOWN_ISSUES-AMD.md` |
| `gfx1100`, `gfx1101`, `gfx1102` | Buildable | RDNA 3 dGPU; untested |
| `gfx1030` | Buildable | RDNA 2; untested |

| `gfx942`, `gfx950` | Buildable | Instinct (CDNA 3, wave64); see [Instinct targets](#instinct-targets) |
| `gfx1036` | Buildable | Raphael iGPU; runtime defects in `KNOWN_ISSUES-AMD.md` |

If you run on something other than gfx1151 — pass or fail — please open an issue.

Confirm your GPU:

```bash
rocminfo | grep -E 'Name:|gfx'
# or, if rocminfo is not on PATH:
find /opt/rocm -name rocminfo -executable 2>/dev/null
```

### Instinct targets

The Instinct parts (`gfx942`: MI300A/MI300X/MI325X, `gfx950`: MI350X/MI355X)
build with the same script, one or several architectures per library:

```bash
HIP_ARCH=gfx942,gfx950 ./build_amd.sh
```

`docker/rocm/Dockerfile` has one image per target (`--target warp_gfx942` on
ROCm 7.2, `--target warp_gfx950` on ROCm 10.1.0); see
[Instinct images](#instinct-images) below. What differs from gfx1151:

- **Wavefront width.** CDNA runs 64-lane wavefronts; the tile lane masks follow
  the target family (`WP_TILE_WARP_SIZE` is 64 on gfx9).
- **`wp.tile_matmul` uses rocWMMA** where it applies: a 16x16 `float` result,
  `float` operands, and `block_dim=64` (one wavefront per block) run on the
  fp32 16x16x4 MFMA instruction. Other shapes, other block sizes and all RDNA
  targets use the scalar GEMM. The path needs the rocWMMA headers
  (`<rocm>/include/rocwmma`); without them it is compiled out. The
  `"hip_tile_rocwmma"` module option set to `False` builds a module without
  it. Measured numbers are under
  [rocWMMA `tile_matmul` timing](#rocwmma-tile_matmul-timing).
- **No textures.** CDNA devices have no image support, so `wp.Texture1D/2D/3D`
  cannot be created there. `Device.is_texture_supported` is `False`, creation
  raises a `RuntimeError` saying so, and the texture tests skip.
- **Architecture names carry target features** (`gfx950:sramecc+:xnack-`).
  Kernel-cache file names spell the colons as `_`.

ROCm installed from the TheRock pip wheels (`rocm[devel]`, which puts the SDK
in `site-packages/_rocm_sdk_devel` and leaves `ROCM_PATH` unset) is found
without configuration: kernel compiles look for the ROCm headers under
`ROCM_PATH`/`ROCM_HOME`/`HIP_PATH`, `/opt/rocm`, those wheel packages, and next
to a `hipcc` on `PATH`, in that order.

On Instinct hardware this has been run on one MI355X node (gfx950, ROCm 7.15
on the host): the build, the rocWMMA tests and timing below, and both images.
gfx942 hardware is untested.

#### rocWMMA `tile_matmul` timing

`tools/bench_tile_rocwmma.py` builds each kernel twice, with and without the
rocWMMA path (the `"hip_tile_rocwmma"` module option), and times both on the
same inputs. One tile is one 64-thread block: load a 16xK and a Kx16 tile,
multiply into a 16x16 tile, store. MI355X (gfx950), ROCm 7.15, 262144 tiles
per launch, median of 15 synchronized samples of at least 0.2 s each:

| Kernel | rocWMMA | Scalar GEMM | Speedup |
|---|---|---|---|
| 16x16, K=16: load, multiply, store | 0.137 ms | 0.199 ms | 1.46x |
| 16x16, K=32: load, multiply, store | 0.223 ms | 0.344 ms | 1.54x |
| 16x16, K=64: load, multiply, store | 0.409 ms | 0.640 ms | 1.56x |
| 16x16, K=16, `alpha`/`beta`: load, multiply, store | 0.179 ms | 0.272 ms | 1.52x |
| 16x16, K=16: 64 multiplies per tile | 47.0 GFLOP/s | 13.0 GFLOP/s | 3.62x |
| 16x16, K=32: 64 multiplies per tile | 32.6 GFLOP/s | 14.2 GFLOP/s | 2.30x |
| 16x16, K=64: 64 multiplies per tile | 34.6 GFLOP/s | 14.8 GFLOP/s | 2.33x |
| 16x16, K=16, `alpha`/`beta`: 64 multiplies per tile | 47.1 GFLOP/s | 13.0 GFLOP/s | 3.63x |

The quartiles are within 1% of the medians except K=32, where the upper
quartile is up to 11% above. At 16384 and 1048576 tiles the ratios are
similar (1.45x to 1.66x, and 2.2x to 3.7x for 64 multiplies). At 1024 tiles a
single multiply per tile is lost in the launch overhead (4-5 µs either way).
The two builds return bit-identical results in every case.

Shapes and block sizes outside the path (16x16 at `block_dim=128`, 32x32,
64x64) compile to the same instructions with the option on or off.

#### Instinct images

Build from the repository root; `--network=host` is only needed where the
default bridge has no route out:

```bash
docker build --network=host -f docker/rocm/Dockerfile --target warp_gfx950 -t warp-rocm:gfx950 .
docker build --network=host -f docker/rocm/Dockerfile --target warp_gfx942 -t warp-rocm:gfx942 .
```

The gfx942 target takes other architectures through
`--build-arg HIP_ARCH=gfx942,gfx950`. Both images include the rocWMMA headers
and a ROCm build of PyTorch, and are about 40 GB each.

Run with the GPU device nodes passed through. Give the render group by the
host's numeric id: the gfx950 image has no `render` group and the gfx942
image's has a different id.

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add "$(getent group render | cut -d: -f3)" \
  warp-rocm:gfx950 \
  bash -c 'python tools/run_gfx1151_smoke.py && python -m unittest warp.tests.cuda.test_tile_rocwmma'
```

`-e HIP_VISIBLE_DEVICES=0` selects a GPU. The smoke test warns that it is not
on gfx1151 and passes. Both images pass this on an MI355X, the gfx942 image
when built with `HIP_ARCH=gfx942,gfx950`.

## Choosing a ROCm version

**Recommended: ROCm 7.14.1.** ROCm 7.14.0 and 7.2.x are also validated; 7.2.x
remains necessary for one workload class below.

The two releases fail in opposite directions:

**ROCm 7.2.x does not reclaim memory allocated inside a captured graph.** Every
graph replay leaks, so sustained simulation degrades badly: the ANYmal C
walking example loses most of its throughput within a few hundred frames and
keeps degrading. On ROCm 7.14 the same example holds a steady rate.

**ROCm 7.14 fails small device allocations under capture-heavy load.**
`hipMalloc` returns NULL for small requests while most device memory is free,
which makes several Newton test modules fail.

For sustained simulation and reinforcement-learning rollouts, use 7.14.1. For
workloads that trip the allocation failure, 7.2.x remains necessary. Both are
described in `KNOWN_ISSUES-AMD.md`.

Two 7.14 specifics:

- The runtime libraries live under `/opt/rocm/core-7.14/lib` behind an
  `/etc/alternatives` symlink chain, and no `ld.so.conf.d` entry is installed,
  so `warp.so` does not load until that directory is on `LD_LIBRARY_PATH`
  (command under [Prerequisites](#prerequisites)).
- A native 7.14 install is not required: the validated configuration runs the
  7.14 userspace in a container over the host's existing kernel driver
  (7.2.1-era driver confirmed working underneath). See
  [ROCm 7.14 in a container](#rocm-714-in-a-container).

### ROCm 7.14 in a container

The validated container is `rocm/dev-ubuntu-24.04:7.14.1-full` under rootless
podman with GPU passthrough. Build from the repository checkout:

```bash
podman run --rm \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined \
  -v "$PWD":/work rocm/dev-ubuntu-24.04:7.14.1-full \
  bash -c 'export PIP_BREAK_SYSTEM_PACKAGES=1 && \
           pip install numpy setuptools packaging wheel && \
           cd /work && python3 build_lib.py --no-cuda --rocm-path=/opt/rocm --hip-arch=gfx1151 --quick'
```

The image ships `pip` but not `python3.12-venv`; with `--rm`, packages
installed into the container filesystem are gone on exit, so for repeated use
install them onto the mounted volume with `pip install --target` and put that
directory on `PYTHONPATH`. To run the resulting `warp.so` (in or out of the
container) set `LD_LIBRARY_PATH` as under [Prerequisites](#prerequisites).

The container's ROCm tree can also be installed natively, side by side with an
existing apt-installed ROCm: copy `/opt/rocm/core-7.14` out of the image to
`/opt/rocm-7.14.1`, register it with
`update-alternatives --install /opt/rocm rocm /opt/rocm-7.14.1 <priority>`,
and point one `ld.so.conf.d` entry at `/opt/rocm/lib` (replacing any entry
pinned to another version — two registered versions in the linker path let a
process mix libraries). This layout is validated: the host build, smoke test,
and Newton examples run against it with no environment variables, and versions
switch with `update-alternatives --set rocm /opt/rocm-<version>` plus
`ldconfig`.

## Tested configuration

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.3 |
| Kernel | 7.0.0-34-generic, in-tree `amdgpu` (7.0.0-28-generic and 6.17.0-1017-oem also validated; some other kernels are known bad — see [Troubleshooting](#troubleshooting)) |
| GPU | AMD Radeon 8060S (gfx1151), 96 GiB unified memory |
| ROCm | 7.14.1 (recommended), installed natively at `/opt/rocm-7.14.1` from the `rocm/dev-ubuntu-24.04:7.14.1-full` image and selected via `update-alternatives`; the same image also validated as a container. 7.14.0 and 7.2.1 validated earlier. **See [Choosing a ROCm version](#choosing-a-rocm-version).** |
| Newton | 1.5.1 from PyPI (`newton[sim,importers,examples]`) |
| MuJoCo | 3.11.0 |
| MuJoCo-Warp | 3.11.0 |
| usd-core | 26.3 (Newton 1.5.1 requires `>=25.5,<26.5`) |
| PyTorch | 2.10.0+rocm7.13.0a20260513, the newest build on `https://rocm.nightlies.amd.com/v2/gfx1151/`, on the ROCm 7.14.1 host. Not usable inside the 7.14.0 container: the wheel bundles a rocm-sdk whose `libamd_comgr` conflicts with the container's LLVM. |
| Python | 3.12 (3.12.3 tested; 3.11 also tested) |
| GCC | 13 (13.3.0 tested) |

### Kernel selection

The in-tree `amdgpu` module must work correctly on gfx1151; not every
kernel's does.

| Kernel | Status |
|---|---|
| `7.0.0-34-generic` | Validated (current test configuration) |
| `7.0.0-28-generic` | Validated |
| `6.17.0-1017-oem` (srcversion `FC7DA320ED9D733CA6A3F1E`) | Validated |
| `6.17.0-1020-oem` | Known bad: page-faults on the first GPU dispatch (CPF MAPPING_ERROR / PERMISSION_FAULTS), hanging the process at 100% CPU |
| `amdgpu-dkms` 30.30 (6.16.x) and 31.20 (6.19.x) series | Known bad: same fault class |

See [Troubleshooting](#troubleshooting) for symptoms and recovery. If your
distribution's update mechanism can replace a working kernel (observed with
Ubuntu OEM kernels via `unattended-upgrade`), hold the kernel and its
meta-package:

```bash
sudo apt-mark hold \
    linux-image-6.17.0-1017-oem \
    linux-modules-6.17.0-1017-oem \
    linux-headers-6.17.0-1017-oem \
    linux-image-oem-24.04d \
    linux-oem-24.04c \
    linux-oem-24.04d \
    linux-headers-oem-24.04d
```

The meta-package (`linux-image-oem-24.04d` or whichever your system
tracks) is the critical one — that is what `unattended-upgrade`
follows. Confirm the holds with `apt-mark showhold | grep linux`.

## Prerequisites

- ROCm at `/opt/rocm`. Recommended: 7.14.1, natively or via
  [the container](#rocm-714-in-a-container); 7.14.0, 7.2.1 and 7.2.4 also validated.
  See [Choosing a ROCm version](#choosing-a-rocm-version).
  **On 7.14 the runtime libraries moved** to `/opt/rocm/core-<version>/lib`
  with no `ld.so.conf.d` entry installed, so `warp.so` will not load until that
  directory is on `LD_LIBRARY_PATH`:
  ```bash
  export LD_LIBRARY_PATH="$(dirname "$(find -L /opt/rocm -maxdepth 3 -name 'libamdhip64.so.7' | head -1)")":$LD_LIBRARY_PATH
  ```
- Linux kernel with the **in-tree** `amdgpu` driver. Do **not** install `amdgpu-dkms` — see [Troubleshooting](#troubleshooting)
- GCC 13
- Python 3.11 or 3.12 with `numpy`, `setuptools`, `packaging`, `wheel`
- `git-lfs`, installed before cloning (`git lfs install`) — the USD/NVDB test
  and example assets are LFS objects; without it the clone contains pointer
  files and anything referencing those assets fails.
- Optional: PyTorch built with ROCm (`torch>=2.11.0+rocm7.2`), only for Warp's
  torch interop. Not required for Warp or Newton.

## Environment setup

You need **Python 3.11 or 3.12** in an isolated environment. Conda is not
required: if you already have one of those Python versions, a plain `venv` is
enough — skip conda entirely:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install numpy setuptools packaging wheel
```

Conda is recommended mainly when you **don't** have the right Python, since it
provides both a pinned interpreter and the environment in one step (`pyenv` or
`uv python install 3.11` work too). Example with conda:

```bash
conda create -n warp-amd python=3.11 -y
conda activate warp-amd

# PyTorch built for gfx1151, from AMD's gfx1151 index (the stock
# download.pytorch.org ROCm wheels do not carry gfx1151). Rolling nightly
# index: you get the current build.
pip install torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/

# Build dependencies
pip install numpy setuptools packaging wheel

# After Warp is built, install it (see Build below):
# pip install -e .

# For Newton physics examples, see the pinned stack under
# [Newton integration](#newton-integration).
```

PyTorch is optional (Warp's torch interop and nothing else requires it). If it
is installed, import `torch` before `warp` in any process so the correct ROCm
runtime is loaded first.

> **This PyTorch build is GPU-accelerated on the iGPU** — it is not a CPU-only
> build. ROCm PyTorch reuses PyTorch's **CUDA** namespace for the AMD GPU, so
> `torch.device("cuda")`, `torch.cuda.is_available()`, and `tensor.to("cuda")`
> all target the gfx1151 iGPU via HIP (this is *not* NVIDIA CUDA). Verify:
>
> ```bash
> python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.version.hip)"
> # -> True AMD Radeon 8060S Graphics 7.2.xxxxx
> ```
>
> If `is_available()` is `False` or `torch.version.hip` is `None`, the wrong
> (CPU-only or non-gfx1151) wheel is installed — reinstall from the gfx1151
> index above.

## Build

```bash
./build_amd.sh
```

Environment variables:

- `HIP_ARCH` — comma-separated target architectures (default: `gfx1151`)
- `ROCM_PATH` — ROCm install path (default: `/opt/rocm`)

Equivalent explicit invocation:

```bash
python build_lib.py --no-cuda --rocm-path=/opt/rocm --hip-arch=gfx1151 --quick
```

Output: `warp/bin/warp.so` and `warp/bin/warp-clang.so` (the CPU kernel
compiler). The compile takes about 1-2 minutes per architecture at the default
8 jobs on a modern desktop CPU; the first-ever
build additionally downloads the LLVM/Clang toolchain dependency.

## Install

```bash
pip install -e .
```

## Smoke test

```bash
python tools/run_gfx1151_smoke.py
```

Exercises kernel compilation, device memory allocation, and two elementwise
kernel launches. On success the output includes `✅ GFX1151 smoke test PASSED`.
It does not exercise graph capture, BVH/mesh queries, or sustained execution;
for those classes see `KNOWN_ISSUES-AMD.md`.

## Newton integration

Install stock Newton 1.5.1 with its simulation, importer and example extras
into the same environment; this pulls the matching MuJoCo stack:

```bash
pip install "newton[sim,importers,examples]==1.5.1"
# resolves to mujoco 3.11.0, mujoco-warp 3.11.0, usd-core 26.3, warp-nn 0.3.x
```

Newton depends on `warp-lang`; because this checkout is already installed
editable (Build and Install above) and satisfies that requirement, `pip` keeps
it rather than fetching the stock wheel. Check that the port is the Warp that
gets imported: `python -c "import warp; print(warp.__file__)"` must print a
path inside this checkout.

Conditional graph nodes are emulated on HIP (see `KNOWN_ISSUES-AMD.md`), so
Newton's MuJoCo solver uses its default graph-conditional path. No wrapper or
patch is needed.

The `patches/newton/` directory targets Newton 1.0.0 and does not apply to
1.5.1.

### Verifying Newton works

Five robot examples (`robot_anymal_c_walk`, `robot_cartpole`, `robot_h1`, `robot_g1`,
`robot_allegro_hand`) run to completion on every validation pass of this port with stock
Newton 1.5.1. Examples outside this set are covered by the platform
classes in `KNOWN_ISSUES-AMD.md`.

Newton's benchmark mode prints its own sustained rate. Run a long frame
count: on ROCm 7.2.x, graph-replaying examples decay with runtime instead of
holding a steady rate — see
[Choosing a ROCm version](#choosing-a-rocm-version).

```bash
# ANYmal C walk (MuJoCo solver)
python -m newton.examples robot_anymal_c_walk --viewer null --benchmark --num-frames 1200

# cable_twist (no MuJoCo)
python -m newton.examples cable_twist --viewer null --benchmark --num-frames 1200
```

## Troubleshooting

### `Warp must be built with CUDA Toolkit 12.4+ to enable conditional graph nodes`

The imported Warp predates conditional graph emulation on HIP. Build and
install this branch; `python -c "import warp; print(warp.is_conditional_graph_supported())"`
prints `True`.

### `hsa_queue_create` page fault / hang at first kernel launch

Two causes share the same symptom: first GPU op hangs at 100% CPU, GPU
idle on `rocm-smi`, kernel log shows a CPF (Command Processor Fetch)
page fault — typically `GCVM_L2_PROTECTION_FAULT_STATUS` with
`MAPPING_ERROR: 0x1`, `PERMISSION_FAULTS: 0x3`.

**Cause A — `amdgpu-dkms` overriding the in-tree module.**
The DKMS packages in the 30.30 (6.16.x) and 31.20 (6.19.x) series
trigger this on gfx1151 (ROCm issue #6118). Fix:

```bash
sudo apt remove amdgpu-dkms
sudo reboot
```

Keep `amdgpu-dkms-firmware` — it only provides firmware blobs and is safe.

**Cause B — newer in-tree kernel (Ubuntu OEM kernel updates).**
The Ubuntu OEM kernel `6.17.0-1020-oem` (and likely later) ships an
in-tree `amdgpu` with a different srcversion (`0AA0C630...` vs the
known-good `FC7DA320...` in 1017). It hits the same fault class on the
first GPU dispatch even though no DKMS is involved.
`unattended-upgrade` can replace 1017 with 1020.

Fix: reboot into the known-good kernel via GRUB, then hold it.

```bash
# Find the entry
sudo awk -F"'" '/menuentry / {print $2}' /boot/grub/grub.cfg | grep 1017

# Set next-boot only (not permanent)
sudo grub-reboot 'Advanced options for Ubuntu>Ubuntu, with Linux 6.17.0-1017-oem'
sudo reboot

# After login, pin so the meta-package can't move it again
sudo apt-mark hold \
    linux-image-6.17.0-1017-oem linux-modules-6.17.0-1017-oem linux-headers-6.17.0-1017-oem \
    linux-image-oem-24.04d linux-oem-24.04c linux-oem-24.04d linux-headers-oem-24.04d
```

The meta-package name (`linux-image-oem-24.04d` above) varies; check
`apt-cache rdepends --installed linux-image-6.17.0-<current>-oem` to
find which meta is pulling new kernels in.

### Confirming which `amdgpu` module is loaded

```bash
modinfo amdgpu | grep -E 'filename|srcversion'
```

A working gfx1151 setup loads the in-tree module from your distribution kernel — the `filename` should be under `/lib/modules/$(uname -r)/kernel/...`, not `.../updates/dkms/`. The known-good srcversion for `6.17.0-1017-oem` is `FC7DA320ED9D733CA6A3F1E`.

### `HSA_STATUS_ERROR_EXCEPTION code 0x1016` in cloth/softbody examples

Historically an illegal-opcode trap from sub-warp (`block_dim < 32`)
`wp.tile_reduce` in Newton VBD/Style3D kernels. It does **not** occur on this
build with ROCm 7.2 — verified that ROCm 7.2 runs one workgroup per wave, so the
reduction is correctly scoped and no patch is needed. If you hit it on an older
ROCm, pad those launches to a full wave (32) with passenger-thread gating.

### Segfault during `wp.capture_launch` on long differentiable simulations

This was an HIP runtime stack overflow during `hipGraphInstantiate` on long
graph dependency chains. It is fixed in this build via a worker-thread
workaround inside `wp_cuda_graph_create_exec`. If you still see segfaults,
verify you are running the patched `warp.so` (not a stale install).

### Diffsim examples appear to hang (`diffsim_bear`, etc.)

They are not hung — differentiable-simulation examples JIT-compile very
large forward+adjoint tile kernels, and HIPRTC (which cannot use
precompiled headers) takes **~15 minutes on the first run** on gfx1151.
The kernel cache makes subsequent runs skip this entirely. `diffsim_bear`
then takes ~10 minutes of actual training on the iGPU. If you need to
demo one, prewarm the cache first (run it once with `--viewer null`).
Watch progress by listing new module directories in the kernel cache
(`ls -lat ~/.cache/warp/<version>/ | head`) — if new entries keep
appearing, compilation is progressing.

### `libmathdx` errors

`libmathdx` is unsupported on HIP builds. There is no workaround on AMD currently.

### `texture_sample_*` returns zero

Fixed. The HIP device branches of ``texture_sample_helper`` were stubs;
they now call the ``tex1D/tex2D/tex3D`` intrinsics (HIPRTC compiles them
fine on ROCm 7.2). Validated bit-exact against the CPU sampler on gfx1151.
If you still see zeros, the kernel using ``texture_sample`` may have a
stale cached cubin compiled against the old stub header — Warp's module
hash does not include native headers, so touch the kernel source (or use
a fresh ``WARP_CACHE_PATH``) to force a recompile.

## Optional settings

The defaults are the validated configuration: the fast float atomics are on, the others off.

| Setting | Effect |
|---|---|
| `WARP_HIP_FAST_FP_ATOMICS=0` | Restore the compare-and-swap retry loop HIP emits for float32 atomic accumulation; by default it is lowered to the hardware atomic. Also available as `warp.config.hip_fast_fp_atomics = False` or the `"hip_fast_fp_atomics"` module option set to `False`. |
| `WARP_ENABLE_UMA_HYBRID=1` | Serve allocations of 64 KB or more from unified (managed) memory for zero-copy CPU access. |
| `WARP_HIP_GRAPH_FREE_NODES=1` | Enable graph memory-free nodes. |
| `WARP_HIP_USE_ASYNC_POOL=1` | Use the stream-ordered pool for all allocations rather than only during graph capture. |
| `WARP_HIP_ALLOC_CACHE=1` | Reuse freed device memory instead of returning each block with a plain free, which costs the device. Freed blocks wait until one device synchronize shows that no queued work uses them, then serve new allocations of the same size; up to 2 GiB is kept. Helps workloads that free many temporaries per step, such as MuJoCo-Warp through Newton. The synchronize invalidates a graph capture that another library runs at the same time in relaxed mode on another stream or in thread-local mode on another thread (for example `torch.cuda.graph` with `capture_error_mode="relaxed"` or `"thread_local"`); a plain free would leave it intact. |
| `WARP_MEMPOOL_RELEASE_THRESHOLD=0.5` | Let each device's memory pool keep freed memory, here up to half of the device's memory, instead of returning it to the driver at every synchronization. A workload that frees temporaries on every step and synchronizes in between, such as MuJoCo-Warp through Newton, otherwise pays that release at each synchronization. Also available as `warp.config.mempool_release_threshold`; not specific to HIP. |

`WARP_HIP_FAST_FP_ATOMICS` matters if a kernel funnels a reduction through one
atomic address, where HIP's own lowering is slow — see
[Single-address atomic contention](KNOWN_ISSUES-AMD.md#single-address-atomic-contention).
It is safe on Warp's own allocations, including the UMA hybrid allocator and
`ReadbackBuffer`, which are marked coarse-grained so the hardware instruction
works on them. It is **not** safe on managed memory Warp did not allocate that
way, such as `CudaManagedAllocator` arrays: the hardware atomic silently discards
updates to host-coherent memory.

The others change allocation or graph behaviour that the validated baseline
does not use. `WARP_HIP_GRAPH_FREE_NODES=1` in particular fixes a small number of
examples and breaks a larger number — see
[KNOWN_ISSUES-AMD.md](KNOWN_ISSUES-AMD.md).

## Limitations

- `libmathdx` is unsupported on HIP builds.
- Single-architecture compile time: about 1-2 minutes, plus a one-time LLVM/Clang toolchain download on the first build.
- Validated only on gfx1151. Other RDNA architectures are buildable but untested.

## License

Apache-2.0. See `LICENSE.md`.
