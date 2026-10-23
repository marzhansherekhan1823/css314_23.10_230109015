"""Task 1 - Hardware Probe & Serial Baseline

Run:  python task1.py | tee task1_output.txt
"""
import os

# Force any hidden native threading (BLAS/OpenMP) to a single thread BEFORE
# importing numpy, so the computation is strictly serial.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import platform
import time

import numpy as np

STUDENT_ID = "230109015"   # <-- replace with your Student ID (digits only is best)
N = 20_000_000


def seed_from_id(student_id: str) -> int:
    """Deterministic integer seed derived from the Student ID."""
    digits = "".join(ch for ch in student_id if ch.isdigit())
    if digits:
        return int(digits) % (2**32)
    return sum(ord(c) * 31 ** i for i, c in enumerate(student_id)) % (2**32)


def cpu_model() -> str:
    """Best-effort CPU model string across Linux / macOS / Windows."""
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            import subprocess
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def main():
    seed = seed_from_id(STUDENT_ID)

    print(f"Student ID       : {STUDENT_ID}")
    print(f"Seed used        : {seed}")
    print(f"OS               : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Platform         : {platform.platform()}")
    print(f"CPU model        : {cpu_model()}")
    print(f"Logical cores    : {os.cpu_count()}")
    print(f"Python / NumPy   : {platform.python_version()} / {np.__version__}")
    print(f"Array size       : {N:,} float64 elements")

    # --- Data generation (reported separately, not part of the compute time) ---
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    data = rng.random(N, dtype=np.float64)
    gen_ms = (time.perf_counter() - t0) * 1000.0

    # --- Serial computation: sum + standard deviation (single thread) ---
    t1 = time.perf_counter()
    total = data.sum()
    mean = total / N
    # population standard deviation, computed in a second serial pass
    std = np.sqrt(np.sum((data - mean) ** 2) / N)
    compute_ms = (time.perf_counter() - t1) * 1000.0

    print(f"Sum              : {total:.6f}")
    print(f"Std deviation    : {std:.9f}")
    print(f"Generation time  : {gen_ms:.2f} ms")
    print(f"Serial compute   : {compute_ms:.2f} ms (wall-clock, sum + std)")


if __name__ == "__main__":
    main()
