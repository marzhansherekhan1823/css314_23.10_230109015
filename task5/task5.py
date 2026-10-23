"""Task 5 - The Overhead Crossover Point

Run:  python task5.py | Out-File -Encoding utf8 task5_output.txt   (PowerShell)
      python task5.py | tee task5_output.txt                        (Linux/macOS)
      python task5.py 8            (optional: override worker count)

Kernel: y[i] = a * x[i] + b   (float64, in-place into a preallocated y)

Serial   : one NumPy call over the whole array, main thread only.
Parallel : persistent ThreadPoolExecutor with all logical cores; the array is split
           into contiguous chunks (zero-copy views) and every chunk is processed by
           one worker (NumPy releases the GIL). Per-call cost INCLUDES task
           submission, thread wake-up and the join - that is the overhead being
           measured. The pool itself is created once, outside the timed region.

The script finds the crossover automatically: first on the required sizes, then it
refines the bracket with a finer geometric sweep between the last "slower" size and
the first "faster" size.
"""
import math
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np

STUDENT_ID = "230109015"
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
A, B = 2.5, 1.0
MIN_BATCH_S = 0.03      # each timed batch runs at least this long
REPEATS = 7             # batches per measurement; median per-call time is used
REFINE_POINTS = 10      # extra sizes inside the bracket


def seed_from_id(student_id):
    digits = "".join(c for c in student_id if c.isdigit())
    return int(digits) % (2**32) if digits else 12345


def serial_kernel(x, y):
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def chunk_kernel(x, y, lo, hi):
    ys = y[lo:hi]
    np.multiply(x[lo:hi], A, out=ys)
    np.add(ys, B, out=ys)


def make_parallel(pool, workers, x, y):
    bounds = np.linspace(0, len(x), workers + 1, dtype=np.int64).tolist()
    spans = [(bounds[i], bounds[i + 1]) for i in range(workers)]

    def run():
        futs = [pool.submit(chunk_kernel, x, y, lo, hi) for lo, hi in spans]
        wait(futs)
        for f in futs:          # surface any worker exception
            f.result()
    return run


def per_call_seconds(fn):
    """Median per-call wall time, batching calls so timer resolution is irrelevant."""
    fn()                                                    # warm-up (page faults, caches)
    t0 = time.perf_counter()
    fn()
    single = max(time.perf_counter() - t0, 1e-9)
    reps = max(1, math.ceil(MIN_BATCH_S / single))
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        samples.append((time.perf_counter() - t0) / reps)
    return statistics.median(samples)


def measure(n, pool, workers, rng):
    x = rng.random(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    ts = per_call_seconds(lambda: serial_kernel(x, y))
    expected = A * x + B
    tp = per_call_seconds(make_parallel(pool, workers, x, y))
    assert np.allclose(y, expected), "parallel result mismatch"
    return ts * 1e6, tp * 1e6          # microseconds


def find_crossover(rows):
    """rows: sorted list of (n, serial_us, parallel_us).
    Crossover = smallest tested N such that parallel is faster for it AND every larger N.
    Returns (status, n_slower, n_faster)."""
    faster = [par < ser for _, ser, par in rows]
    if not any(faster[-1:]):
        return "never", rows[-1][0], None
    k = len(rows) - 1
    while k > 0 and faster[k - 1]:
        k -= 1
    if k == 0:
        return "always", None, rows[0][0]
    return "found", rows[k - 1][0], rows[k][0]


def print_table(rows):
    header = f"{'N':>12} | {'Serial (us)':>13} | {'Parallel (us)':>13} | {'Speedup':>8} | Winner"
    line = "-" * (len(header) + 8)
    print(line)
    print(header)
    print(line)
    for n, ser, par in rows:
        sp = ser / par
        print(f"{n:>12,} | {ser:>13.2f} | {par:>13.2f} | {sp:>7.2f}x | "
              f"{'PARALLEL' if par < ser else 'serial'}")
    print(line)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else (os.cpu_count() or 1)

    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS / Platform : {platform.platform()}")
    print(f"Logical cores : {os.cpu_count()}  |  Workers used: {workers}")
    print(f"Python / NumPy: {platform.python_version()} / {np.__version__}")
    print(f"Kernel        : y[i] = {A} * x[i] + {B}  (float64)")
    print(f"Parallel model: persistent ThreadPoolExecutor, {workers} workers, contiguous chunks")
    print(f"Timing        : median of {REPEATS} batches, per-call time, batches >= {MIN_BATCH_S*1000:.0f} ms\n")

    rng = np.random.default_rng(seed_from_id(STUDENT_ID))
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Phase 1: required sizes
        for n in SIZES:
            results[n] = measure(n, pool, workers, rng)
        rows = [(n, *results[n]) for n in sorted(results)]
        print("PHASE 1 - required sizes")
        print_table(rows)
        status, lo, hi = find_crossover(rows)
        print(f"Coarse result: {describe(status, lo, hi)}\n")

        # Phase 2: refine the bracket with a finer geometric sweep
        if status == "found":
            fine = sorted({int(round(math.exp(v))) for v in
                           np.linspace(math.log(lo), math.log(hi), REFINE_POINTS + 2)[1:-1]})
            for n in fine:
                if n not in results:
                    results[n] = measure(n, pool, workers, rng)
            rows = [(n, *results[n]) for n in sorted(results)]
            print(f"PHASE 2 - refinement between N={lo:,} and N={hi:,}")
            print_table([r for r in rows if lo <= r[0] <= hi])
            status, lo, hi = find_crossover(rows)

    print("=" * 72)
    print("CROSSOVER RESULT")
    print(describe(status, lo, hi))
    if status == "found":
        mid = int(round(math.sqrt(lo * hi)))
        print(f"Best estimate of break-even N (geometric midpoint): ~{mid:,} elements")
    print("=" * 72)


def describe(status, lo, hi):
    if status == "found":
        return (f"Parallel is SLOWER than serial up to N = {lo:,} and FASTER from "
                f"N = {hi:,} upward.\n  Crossover lies in the range ({lo:,}, {hi:,}].")
    if status == "always":
        return f"Parallel is faster at every tested size (crossover is below N = {hi:,})."
    return f"Parallel never beat serial up to N = {lo:,} on this machine."


if __name__ == "__main__":
    main()
