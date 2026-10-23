"""Task 7 - Arithmetic Intensity vs. Memory Bottlenecks

Run:  python task7.py | Out-File -Encoding utf8 task7_output.txt   (PowerShell)
      python task7.py | tee task7_output.txt                        (Linux/macOS)
      python task7.py 5000000      (optional: smaller N for a quick test)

Kernel A (memory-bound)  : B[i] = A[i]                              (copy: ~0 FLOPs per byte moved)
Kernel B (compute-bound) : B[i] = sin(A[i]) * cos(A[i]) + tan(A[i]) (many FLOPs per byte moved)

Parallel model: persistent ThreadPoolExecutor, array split into contiguous chunks (zero-copy
views). NumPy releases the GIL inside copyto / sin / cos / tan, so threads genuinely overlap.
Kernel B works in small cache-resident sub-blocks with per-worker scratch buffers, so its
temporaries never go to RAM: its traffic stays at one read + one write per element, and its
runtime is dominated by the math, not the memory bus.

Efficiency = Speedup / Cores, with Speedup = T(1) / T(p).
"""
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = "230109015"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30_000_000
BLOCK = 16_384            # elements per cache-resident sub-block for kernel B (128 KB / array)
REPEATS_A = 7
REPEATS_B = 3
STALL_EFF = 0.50          # efficiency at max cores below this ...
STALL_GAIN = 1.25         # ... and gain from the previous core count below this => "stalled"


def seed_from_id(student_id):
    digits = "".join(c for c in student_id if c.isdigit())
    return int(digits) % (2**32) if digits else 12345


# ---------------------------------------------------------------- kernels
def kernel_a(A, B, lo, hi):
    np.copyto(B[lo:hi], A[lo:hi])


def kernel_b(A, B, lo, hi):
    t1 = np.empty(BLOCK)
    t2 = np.empty(BLOCK)
    for s in range(lo, hi, BLOCK):
        e = min(s + BLOCK, hi)
        n = e - s
        a = A[s:e]
        x1, x2 = t1[:n], t2[:n]
        np.sin(a, out=x1)
        np.cos(a, out=x2)
        np.multiply(x1, x2, out=x1)
        np.tan(a, out=x2)
        np.add(x1, x2, out=B[s:e])


def run_parallel(pool, kernel, A, B, workers):
    bounds = np.linspace(0, len(A), workers + 1, dtype=np.int64).tolist()
    futs = [pool.submit(kernel, A, B, bounds[i], bounds[i + 1]) for i in range(workers)]
    for f in futs:
        f.result()


def bench(kernel, A, B, workers, repeats):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        run_parallel(pool, kernel, A, B, workers)           # warm-up (threads, page faults)
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            run_parallel(pool, kernel, A, B, workers)
            samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


# ---------------------------------------------------------------- reporting
def table(title, times, bytes_moved):
    t1 = times[min(times)]
    header = (f"{'Cores':>6} | {'Time (ms)':>10} | {'Speedup':>8} | {'Efficiency':>10} | "
              f"{'Eff. BW (GB/s)':>14}")
    line = "-" * len(header)
    print(title)
    print(line)
    print(header)
    print(line)
    stats = {}
    for p in sorted(times):
        t = times[p]
        sp = t1 / t
        eff = sp / p
        gbs = bytes_moved / t / 1e9
        stats[p] = (sp, eff, gbs)
        print(f"{p:>6} | {t * 1000:>10.2f} | {sp:>7.2f}x | {eff * 100:>9.1f}% | {gbs:>14.2f}")
    print(line)
    return stats


def judge(stats, max_cores):
    """Return (stalled, eff_max, gain) where gain = S(max)/S(previous tested count)."""
    counts = sorted(stats)
    pmax = counts[-1]
    eff = stats[pmax][1]
    if len(counts) < 2 or pmax <= 1:
        return False, eff, None
    gain = stats[pmax][0] / stats[counts[-2]][0]
    return (eff < STALL_EFF and gain < STALL_GAIN), eff, gain


def main():
    max_cores = os.cpu_count() or 1
    counts = sorted({1, 2, 4, 8, max_cores} if max_cores >= 2 else {1, 2})
    counts = [c for c in counts if c <= max(max_cores, 2)]

    print(f"Student ID      : {STUDENT_ID}")
    print(f"OS / Platform   : {platform.platform()}")
    print(f"Logical cores   : {max_cores}")
    print(f"Python / NumPy  : {platform.python_version()} / {np.__version__}")
    print(f"Array size      : {N:,} float64 ({N * 8 / 1e6:.0f} MB per array, 2 arrays)")
    print(f"Core counts     : {counts}  (required: 1, 2, max={max_cores}; extra points added)")
    print(f"Repeats         : kernel A x{REPEATS_A}, kernel B x{REPEATS_B} (median)\n")

    rng = np.random.default_rng(seed_from_id(STUDENT_ID))
    A = rng.random(N, dtype=np.float64)
    B = np.empty(N, dtype=np.float64)

    # ---- Kernel A
    times_a = {p: bench(kernel_a, A, B, p, REPEATS_A) for p in counts}
    assert np.array_equal(A, B), "kernel A result mismatch"
    # ---- Kernel B
    times_b = {p: bench(kernel_b, A, B, p, REPEATS_B) for p in counts}
    sample = slice(0, min(N, 100_000))
    ref = np.sin(A[sample]) * np.cos(A[sample]) + np.tan(A[sample])
    assert np.allclose(B[sample], ref), "kernel B result mismatch"

    bytes_moved = 2 * N * 8     # one read of A + one write of B (write-allocate traffic not counted)
    stats_a = table("KERNEL A - Memory-bound: B[i] = A[i]", times_a, bytes_moved)
    print()
    stats_b = table("KERNEL B - Compute-bound: B[i] = sin(A[i])*cos(A[i]) + tan(A[i])", times_b, bytes_moved)

    stalled_a, eff_a, gain_a = judge(stats_a, max_cores)
    stalled_b, eff_b, gain_b = judge(stats_b, max_cores)
    pmax = max(counts)
    peak_a = max(v[2] for v in stats_a.values())
    peak_p = max(stats_a, key=lambda p: stats_a[p][2])

    print("\n" + "=" * 74)
    print("VERDICT")
    print(f"Kernel A (copy)       : efficiency at {pmax} cores = {eff_a * 100:.1f}%"
          + (f", gain from previous core count = {gain_a:.2f}x" if gain_a else ""))
    print(f"Kernel B (sin/cos/tan): efficiency at {pmax} cores = {eff_b * 100:.1f}%"
          + (f", gain from previous core count = {gain_b:.2f}x" if gain_b else ""))
    print(f"Kernel A peak effective bandwidth: {peak_a:.2f} GB/s (reached at {peak_p} cores)")
    if stalled_a and not stalled_b:
        print("\n>>> Kernel A (memory-bound vector copy) STALLED in scaling due to memory channel")
        print("    saturation: adding cores stopped adding bandwidth, so speedup plateaued far below")
        print("    linear. Kernel B (compute-bound) kept scaling because its time is spent in the ALUs,")
        print("    not waiting on the RAM bus.")
    elif stalled_a and stalled_b:
        print("\n>>> BOTH kernels stalled; Kernel A stalled harder"
              if eff_a < eff_b else "\n>>> BOTH kernels stalled; Kernel B stalled harder")
        print("    (check SMT/thermal limits for kernel B; kernel A is the memory-channel-saturated one).")
    elif stalled_b and not stalled_a:
        print("\n>>> Kernel B stalled while Kernel A did not (unexpected: likely SMT/thermal/power limits,")
        print("    not memory bandwidth). No memory channel saturation detected for Kernel A.")
    else:
        print("\n>>> No kernel met the stall criteria on this run (no clear memory channel saturation).")
        print(f"    Efficiency at {pmax} cores: Kernel A {eff_a * 100:.1f}% vs Kernel B {eff_b * 100:.1f}%.")
        if eff_a < eff_b:
            print("    Kernel A scaled worse than Kernel B, consistent with it being the more memory-limited one.")
    print(f"(stall criterion: efficiency at max cores < {STALL_EFF * 100:.0f}% AND gain from previous "
          f"core count < {STALL_GAIN:.2f}x)")
    print("=" * 74)


if __name__ == "__main__":
    main()
