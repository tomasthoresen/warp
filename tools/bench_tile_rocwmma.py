# SPDX-License-Identifier: Apache-2.0

"""Time ``wp.tile_matmul`` on the rocWMMA path against the scalar GEMM.

On the CDNA HIP targets a 16x16 float32 ``wp.tile_matmul`` in a ``block_dim=64``
launch uses the fp32 16x16x4 MFMA instruction through rocWMMA. The
``"hip_tile_rocwmma"`` module option compiles the same kernel without it, so
each case below is built twice and timed on the same inputs.

Every tile is one block: the kernel loads a 16xK and a Kx16 tile, multiplies
them ``reps`` times into a 16x16 tile and stores it. ``reps=1`` is the
load/multiply/store round trip, larger ``reps`` isolates the multiply.

Usage::

    HIP_VISIBLE_DEVICES=0 python tools/bench_tile_rocwmma.py
    python tools/bench_tile_rocwmma.py --tiles 4096 65536 --samples 21 --json out.json
"""

import argparse
import json
import statistics
import time

import numpy as np

import warp as wp


def make_kernel(m, n, k, rocwmma, storage, accumulate):
    """Build ``C[i] = alpha * A[i] @ B[i] + beta * C[i]`` (``accumulate``) or ``C[i] += A[i] @ B[i]``."""

    TILE_M = m
    TILE_N = n
    TILE_K = k
    options = {"hip_tile_rocwmma": rocwmma, "enable_backward": False}

    # The storage argument has to be a literal, hence one definition per variant.
    if accumulate:

        @wp.kernel(module="unique", module_options=options)
        def matmul(
            A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float], reps: int, alpha: float, beta: float
        ):
            i = wp.tid()
            a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0), storage="shared")
            b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(i * TILE_K, 0), storage="shared")
            c = wp.tile_load(C, shape=(TILE_M, TILE_N), offset=(i * TILE_M, 0), storage="shared")
            for _ in range(reps):
                wp.tile_matmul(a, b, c, alpha=alpha, beta=beta)
            wp.tile_store(C, c, offset=(i * TILE_M, 0))

    elif storage == "shared":

        @wp.kernel(module="unique", module_options=options)
        def matmul(
            A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float], reps: int, alpha: float, beta: float
        ):
            i = wp.tid()
            a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0), storage="shared")
            b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(i * TILE_K, 0), storage="shared")
            c = wp.tile_zeros(dtype=float, shape=(TILE_M, TILE_N), storage="shared")
            for _ in range(reps):
                wp.tile_matmul(a, b, c)
            wp.tile_store(C, c, offset=(i * TILE_M, 0))

    else:

        @wp.kernel(module="unique", module_options=options)
        def matmul(
            A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float], reps: int, alpha: float, beta: float
        ):
            i = wp.tid()
            a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0))
            b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(i * TILE_K, 0))
            c = wp.tile_zeros(dtype=float, shape=(TILE_M, TILE_N))
            for _ in range(reps):
                wp.tile_matmul(a, b, c)
            wp.tile_store(C, c, offset=(i * TILE_M, 0))

    return matmul


CASES = {
    # name: (M, N, K, block_dim, storage, accumulate)
    "16x16x16": (16, 16, 16, 64, "shared", False),
    "16x16x16 register": (16, 16, 16, 64, "register", False),
    "16x16x32": (16, 16, 32, 64, "shared", False),
    "16x16x64": (16, 16, 64, 64, "shared", False),
    "16x16x16 alpha/beta": (16, 16, 16, 64, "shared", True),
    # Shapes and block sizes the rocWMMA path does not take: both builds run the scalar GEMM.
    "16x16x16 block_dim=128 (scalar)": (16, 16, 16, 128, "shared", False),
    "32x32x32 (scalar)": (32, 32, 32, 64, "shared", False),
    "64x64x64 block_dim=128 (scalar)": (64, 64, 64, 128, "shared", False),
}

ALPHA = 0.5
BETA = 0.25


def run_case(name, tiles, reps, samples, min_time, device):
    m, n, k, block_dim, storage, accumulate = CASES[name]

    rng = np.random.default_rng(42)
    a_np = rng.standard_normal((tiles * m, k), dtype=np.float32)
    b_np = rng.standard_normal((tiles * k, n), dtype=np.float32)
    # Keep the accumulated values bounded over reps: |alpha| + |beta| < 1 after the first product.
    c_np = rng.standard_normal((tiles * m, n), dtype=np.float32)

    a = wp.array(a_np, dtype=float, device=device)
    b = wp.array(b_np, dtype=float, device=device)

    out = {}
    results = {}
    for label, rocwmma in (("rocwmma", True), ("scalar", False)):
        kernel = make_kernel(m, n, k, rocwmma, storage, accumulate)
        c = wp.array(c_np, dtype=float, device=device)
        cmd = wp.launch_tiled(
            kernel,
            dim=[tiles],
            inputs=[a, b, c, reps, ALPHA, BETA],
            block_dim=block_dim,
            device=device,
            record_cmd=True,
        )

        # The first launch compiles and loads the module; its result is the one compared.
        cmd.launch()
        wp.synchronize_device(device)
        results[label] = c.numpy()

        # Calibrate the number of launches per sample to min_time, then warm up.
        iters = 1
        while True:
            t0 = time.perf_counter()
            for _ in range(iters):
                cmd.launch()
            wp.synchronize_device(device)
            dt = time.perf_counter() - t0
            if dt >= min_time or iters >= 1 << 20:
                break
            iters = max(iters * 2, int(iters * min_time / max(dt, 1e-9)))

        times = []
        for _ in range(samples):
            t0 = time.perf_counter()
            for _ in range(iters):
                cmd.launch()
            wp.synchronize_device(device)
            times.append((time.perf_counter() - t0) / iters)

        times.sort()
        out[label] = {
            "median_s": statistics.median(times),
            "min_s": times[0],
            "q1_s": times[len(times) // 4],
            "q3_s": times[(3 * len(times)) // 4],
            "iters": iters,
        }

    # Reference for the first launch (reps products from the initial C).
    prod = np.einsum("tik,tkj->tij", a_np.reshape(tiles, m, k), b_np.reshape(tiles, k, n), dtype=np.float64)
    ref = c_np.reshape(tiles, m, n).astype(np.float64) if accumulate else np.zeros((tiles, m, n))
    for _ in range(reps):
        ref = ALPHA * prod + BETA * ref if accumulate else ref + prod
    ref = ref.reshape(tiles * m, n)
    scale = np.abs(ref).max()

    out["max_rel_diff_paths"] = float(np.abs(results["rocwmma"] - results["scalar"]).max() / scale)
    out["max_rel_err_rocwmma"] = float(np.abs(results["rocwmma"] - ref).max() / scale)
    out["max_rel_err_scalar"] = float(np.abs(results["scalar"] - ref).max() / scale)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="Warp device (default: the preferred device)")
    parser.add_argument("--tiles", type=int, nargs="+", default=[1024, 16384, 262144])
    parser.add_argument("--reps", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    parser.add_argument("--samples", type=int, default=15, help="timed samples per build")
    parser.add_argument("--min-time", type=float, default=0.2, help="seconds of launches per sample")
    parser.add_argument("--json", default=None, help="also write the results to this file")
    args = parser.parse_args()

    wp.config.quiet = True
    wp.init()
    device = wp.get_device(args.device)
    print(f"device: {device} ({device.name}) arch: {getattr(device, 'arch_str', '') or 'n/a'}")
    print(f"samples: {args.samples} x >= {args.min_time} s of launches, synchronized; median [q1, q3]")
    print()

    header = (
        f"{'case':34s} {'tiles':>7s} {'reps':>4s} {'rocWMMA ms':>22s} {'scalar ms':>22s} "
        f"{'rocWMMA GFLOP/s':>15s} {'scalar GFLOP/s':>14s} {'speedup':>7s} {'rel diff':>8s}"
    )
    print(header)
    print("-" * len(header))

    records = []
    for name in args.cases:
        m, n, k = CASES[name][:3]
        for tiles in args.tiles:
            for reps in args.reps:
                r = run_case(name, tiles, reps, args.samples, args.min_time, device)
                flops = 2.0 * m * n * k * tiles * reps

                def fmt(s):
                    return f"{s['median_s'] * 1e3:8.4f} [{s['q1_s'] * 1e3:.4f}, {s['q3_s'] * 1e3:.4f}]"

                speedup = r["scalar"]["median_s"] / r["rocwmma"]["median_s"]
                print(
                    f"{name:34s} {tiles:7d} {reps:4d} {fmt(r['rocwmma']):>22s} {fmt(r['scalar']):>22s} "
                    f"{flops / r['rocwmma']['median_s'] / 1e9:15.1f} {flops / r['scalar']['median_s'] / 1e9:14.1f} "
                    f"{speedup:7.2f} {r['max_rel_diff_paths']:8.1e}",
                    flush=True,
                )
                records.append({"case": name, "tiles": tiles, "reps": reps, "speedup": speedup, **r})

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": device.name, "records": records}, f, indent=2)


if __name__ == "__main__":
    main()
