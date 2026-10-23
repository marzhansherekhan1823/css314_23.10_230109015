"""Task 1 - Row-Major vs. Column-Major Parallel Cache Traversal

Run:  python task1.py | Out-File -Encoding utf8 task1_output.txt   (PowerShell)
      python task1.py | tee task1_output.txt                        (Linux/macOS)
      python task1.py 4000      (optional: smaller matrix for a quick test)

Matrix: 8,000 x 8,000 float64 (512 MB), C-contiguous (row-major) in memory, seeded from Student ID.

Mode A (row-major)    : each worker owns a block of ROWS; for every row it sums a contiguous
                        8,000-element slice -> linear walk through memory, one cache line serves
                        8 consecutive elements.
Mode B (column-major) : each worker owns a block of COLUMNS; for every column it sums a strided
                        view arr[:, j] -> consecutive elements are 64,000 bytes apart, so every
                        access lands on a different cache line AND a different memory page.

Both modes issue exactly 8,000 NumPy reductions of 8,000 elements each, so interpreter overhead
is identical; the only difference is memory access pattern. NumPy releases the GIL during the
reductions, so worker threads genuinely run in parallel.
"""
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = "230109015"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
REPEATS = 5


def seed_from_id(student_id):
    digits = "".join(c for c in student_id if c.isdigit())
    return int(digits) % (2**32) if digits else 12345


def physical_cores(logical):
    """Best-effort physical core count (Windows / Linux / macOS); fallback logical // 2."""
    try:
        system = platform.system()
        if system == "Windows":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"],
                text=True, timeout=20).strip()
            v = int(out.split()[0])
        elif system == "Linux":
            out = subprocess.check_output(
                ["sh", "-c", "lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l"],
                text=True, timeout=20).strip()
            v = int(out)
        elif system == "Darwin":
            v = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
        else:
            v = 0
        if 0 < v <= logical:
            return v
    except Exception:
        pass
    return max(1, logical // 2)


def split(n, parts):
    bounds = [round(i * n / parts) for i in range(parts + 1)]
    return [(bounds[i], bounds[i + 1]) for i in range(parts)]


def sum_rows(mat, lo, hi):
    total = 0.0
    for i in range(lo, hi):
        total += mat[i].sum()            # contiguous slice
    return total


def sum_cols(mat, lo, hi):
    total = 0.0
    for j in range(lo, hi):
        total += mat[:, j].sum()         # strided slice, stride = N * 8 bytes
    return total


def run_mode(pool, fn, mat, spans):
    t0 = time.perf_counter()
    futs = [pool.submit(fn, mat, lo, hi) for lo, hi in spans]
    total = sum(f.result() for f in futs)
    return (time.perf_counter() - t0) * 1000.0, total


def median_time(pool, fn, mat, spans):
    run_mode(pool, fn, mat, spans)                      # warm-up
    times, total = [], 0.0
    for _ in range(REPEATS):
        ms, total = run_mode(pool, fn, mat, spans)
        times.append(ms)
    return statistics.median(times), total


def main():
    logical = os.cpu_count() or 1
    workers = physical_cores(logical)

    print(f"Student ID            : {STUDENT_ID}")
    print(f"OS / Platform         : {platform.platform()}")
    print(f"Logical core count    : {logical}")
    print(f"Physical cores used   : {workers}")
    print(f"Python / NumPy        : {platform.python_version()} / {np.__version__}")
    print(f"Matrix                : {N:,} x {N:,} float64 ({N * N * 8 / 1e6:.0f} MB), row-major layout")
    print(f"Row stride            : {N * 8:,} bytes between vertically adjacent elements")
    print(f"Repeats               : {REPEATS} per mode (median reported)")

    rng = np.random.default_rng(seed_from_id(STUDENT_ID))
    mat = rng.random((N, N), dtype=np.float64)
    reference = float(mat.sum())
    print(f"Reference sum         : {reference:.6f}\n")

    spans = split(N, workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ms_a, sum_a = median_time(pool, sum_rows, mat, spans)
        ms_b, sum_b = median_time(pool, sum_cols, mat, spans)

    rel_a = abs(sum_a - reference) / reference
    rel_b = abs(sum_b - reference) / reference
    ok = max(rel_a, rel_b) < 1e-9

    line = "-" * 62
    print(line)
    print(f"Mode A (row-major, contiguous)   : {ms_a:>10.2f} ms   sum = {sum_a:.6f}")
    print(f"Mode B (column-major, strided)   : {ms_b:>10.2f} ms   sum = {sum_b:.6f}")
    print(line)
    print(f"Cache penalty factor (B / A)     : {ms_b / ms_a:>10.2f} x")
    print(line)
    print(f"Correctness: both modes match reference (rel. error A={rel_a:.1e}, B={rel_b:.1e}) -> "
          f"{'PASS' if ok else 'CHECK'}")


if __name__ == "__main__":
    main()
