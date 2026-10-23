"""Task 2 - Multi-Core Scaling vs. Amdahl's Law

Run:  python task2.py | Out-File -Encoding utf8 task2_output.txt   (PowerShell)
      python task2.py | tee task2_output.txt                        (Linux/macOS)

Parallel strategy: the 20M-element array is split into p contiguous chunks
(zero-copy views). Each worker THREAD sums its chunk with np.sum, which
releases the GIL, so threads really run in parallel. The partial sums are
then combined in the main thread.
"""
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = "230109015"
N = 20_000_000
P_PARALLEL = 0.95      # parallelizable fraction for Amdahl's Law
REPEATS = 7            # runs per worker count; the median is reported


def seed_from_id(student_id: str) -> int:
    digits = "".join(ch for ch in student_id if ch.isdigit())
    if digits:
        return int(digits) % (2**32)
    return sum(ord(c) * 31 ** i for i, c in enumerate(student_id)) % (2**32)


def amdahl(p: int, parallel_fraction: float = P_PARALLEL) -> float:
    """Theoretical max speedup: S(p) = 1 / ((1 - P) + P / p)."""
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / p)


def make_chunks(data: np.ndarray, workers: int):
    """Split data into `workers` contiguous, near-equal views (no copying)."""
    bounds = np.linspace(0, len(data), workers + 1, dtype=np.int64)
    return [data[bounds[i]:bounds[i + 1]] for i in range(workers)]


def parallel_sum(pool: ThreadPoolExecutor, chunks) -> float:
    partials = pool.map(np.sum, chunks)
    return float(sum(partials))


def bench(data: np.ndarray, workers: int):
    """Return (median wall time in ms, computed sum) for `workers` threads."""
    chunks = make_chunks(data, workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        result = parallel_sum(pool, chunks)           # warm-up (spawns threads)
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            result = parallel_sum(pool, chunks)
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), result


def main():
    max_cores = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, max_cores, max_cores * 2})

    print(f"Student ID      : {STUDENT_ID}")
    print(f"OS / Platform   : {platform.platform()}")
    print(f"Logical cores   : {max_cores}")
    print(f"Python / NumPy  : {platform.python_version()} / {np.__version__}")
    print(f"Array size      : {N:,} float64 elements")
    print(f"Parallel model  : ThreadPoolExecutor + np.sum (GIL released)")
    print(f"Amdahl P        : {P_PARALLEL}")
    print(f"Repeats         : {REPEATS} per worker count (median reported)")

    rng = np.random.default_rng(seed_from_id(STUDENT_ID))
    data = rng.random(N, dtype=np.float64)
    reference = float(data.sum())
    print(f"Reference sum   : {reference:.6f}\n")

    results = []
    for p in worker_counts:
        ms, s = bench(data, p)
        results.append((p, ms, s))

    t1 = next(ms for p, ms, _ in results if p == 1)

    header = f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Amdahl Upper Bound':>18}"
    line = "-" * len(header)
    print(line)
    print(header)
    print(line)
    for p, ms, _ in results:
        print(f"{p:>6} | {ms:>14.2f} | {t1 / ms:>7.2f}x | {amdahl(p):>17.2f}x")
    print(line)

    max_err = max(abs(s - reference) for _, _, s in results)
    print(f"\nMax |parallel sum - reference| = {max_err:.3e}")
    print(f"Amdahl asymptote (p -> inf)    = {1 / (1 - P_PARALLEL):.2f}x")


if __name__ == "__main__":
    main()
