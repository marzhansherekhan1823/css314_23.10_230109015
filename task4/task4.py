"""Task 4 - Work Stealing & Scheduling Under Skew

Run:  python task4.py | Out-File -Encoding utf8 task4_output.txt   (PowerShell)
      python task4.py | tee task4_output.txt                        (Linux/macOS)

1,000 items; item i costs sleep(0.0001 * (i % 20)). 4 workers (threads; sleep
releases the GIL so they genuinely overlap).

Strategies
  A  Static            : equal contiguous blocks, one per worker (250 items each)
  B1 Dynamic, chunk=1  : shared queue, workers pull 1 item at a time
  B5 Dynamic, chunk=5  : shared queue, workers pull 5 items at a time
  C  Work stealing     : per-worker deques (chunks of 5); an idle worker steals
                         a chunk from the tail of another worker's deque

Metrics: total wall time, and the delta between the first and the last worker
finishing. A second, bonus workload ("ramp", cost grows with i) is included
because the required i % 20 pattern is nearly balanced under contiguous blocks.
"""
import os
import platform
import queue
import statistics
import threading
import time
from collections import deque

STUDENT_ID = "230109015"
N_ITEMS = 1000
WORKERS = 4
REPEATS = 3
UNIT = 0.0001  # seconds


def cost_mod20(i):   # required workload
    return UNIT * (i % 20)


def cost_ramp(i):    # bonus workload: heavy tail at high indices
    return UNIT * 2 * (i // 50)


def do_item(cost):
    time.sleep(cost)


# --------------------------------------------------------------------------
# Strategies. Each returns (starts, finishes) lists of perf_counter stamps.
# --------------------------------------------------------------------------
def run_threads(target):
    barrier = threading.Barrier(WORKERS)
    starts, finishes = [0.0] * WORKERS, [0.0] * WORKERS

    def wrapper(w):
        barrier.wait()
        starts[w] = time.perf_counter()
        target(w)
        finishes[w] = time.perf_counter()

    ts = [threading.Thread(target=wrapper, args=(w,)) for w in range(WORKERS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return starts, finishes


def strategy_static(costs):
    per = len(costs) // WORKERS

    def work(w):
        lo = w * per
        hi = len(costs) if w == WORKERS - 1 else lo + per
        for i in range(lo, hi):
            do_item(costs[i])

    return run_threads(work)


def strategy_dynamic(costs, chunk):
    q = queue.Queue()
    for lo in range(0, len(costs), chunk):
        q.put(range(lo, min(lo + chunk, len(costs))))

    def work(w):
        while True:
            try:
                idxs = q.get_nowait()
            except queue.Empty:
                return
            for i in idxs:
                do_item(costs[i])

    return run_threads(work)


def strategy_stealing(costs, chunk=5):
    per = len(costs) // WORKERS
    deques = []
    for w in range(WORKERS):
        lo = w * per
        hi = len(costs) if w == WORKERS - 1 else lo + per
        deques.append(deque(range(a, min(a + chunk, hi)) for a in range(lo, hi, chunk)))
    steals = [0] * WORKERS

    def work(w):
        while True:
            try:
                idxs = deques[w].popleft()           # own work: front
            except IndexError:
                idxs = None
                for off in range(1, WORKERS):        # steal from tail of a victim
                    try:
                        idxs = deques[(w + off) % WORKERS].pop()
                        steals[w] += 1
                        break
                    except IndexError:
                        continue
                if idxs is None:
                    return                           # nothing left anywhere
            for i in idxs:
                do_item(costs[i])

    starts, finishes = run_threads(work)
    return starts, finishes, sum(steals)


# --------------------------------------------------------------------------
def summarize(starts, finishes):
    t0 = min(starts)
    fin = [(f - t0) * 1000.0 for f in finishes]
    return max(fin), min(fin), max(fin) - min(fin), fin


def median_run(fn):
    runs = []
    for _ in range(REPEATS):
        out = fn()
        steals = out[2] if len(out) == 3 else None
        total, first, delta, fin = summarize(out[0], out[1])
        runs.append((total, first, delta, fin, steals))
    runs.sort(key=lambda r: r[0])
    return runs[len(runs) // 2]


def run_workload(name, cost_fn):
    costs = [cost_fn(i) for i in range(N_ITEMS)]
    ideal_ms = sum(costs) / WORKERS * 1000.0
    # exact sleep-model load of each static block, for reference
    per = N_ITEMS // WORKERS
    blocks = [sum(costs[w * per:(w + 1) * per]) * 1000.0 for w in range(WORKERS)]

    print(f"\n=== Workload: {name} ===")
    print(f"Total modeled work : {sum(costs) * 1000:.1f} ms  |  ideal per worker (work/{WORKERS}): {ideal_ms:.1f} ms")
    print("Static block loads : " + ", ".join(f"{b:.1f} ms" for b in blocks))

    strategies = [
        ("A  Static (equal blocks)", lambda: strategy_static(costs)),
        ("B1 Dynamic, chunk=1",      lambda: strategy_dynamic(costs, 1)),
        ("B5 Dynamic, chunk=5",      lambda: strategy_dynamic(costs, 5)),
        ("C  Work stealing, chunk=5", lambda: strategy_stealing(costs, 5)),
    ]
    rows = []
    for label, fn in strategies:
        rows.append((label, *median_run(fn)))

    header = (f"{'Strategy':<26} | {'Total (ms)':>10} | {'First done':>10} | "
              f"{'Last done':>10} | {'Delta (ms)':>10} | {'Efficiency':>10}")
    line = "-" * len(header)
    print(line)
    print(header)
    print(line)
    for label, total, first, delta, fin, steals in rows:
        eff = ideal_ms / total * 100.0
        print(f"{label:<26} | {total:>10.2f} | {first:>10.2f} | {total:>10.2f} | {delta:>10.2f} | {eff:>9.1f}%")
    print(line)
    for label, total, first, delta, fin, steals in rows:
        extra = f"  (steals: {steals})" if steals is not None else ""
        print(f"  {label:<26} per-worker finish (ms): " + ", ".join(f"{f:.1f}" for f in fin) + extra)

    static_total = rows[0][1]
    print("  Speedup vs static: " + ", ".join(
        f"{r[0].split()[0]}={static_total / r[1]:.2f}x" for r in rows[1:]))


def main():
    print(f"Student ID    : {STUDENT_ID}")
    print(f"OS / Platform : {platform.platform()}")
    print(f"Logical cores : {os.cpu_count()}")
    print(f"Python        : {platform.python_version()}")
    print(f"Items         : {N_ITEMS}  |  Workers: {WORKERS}  |  Repeats: {REPEATS} (median by total time)")
    print("Cost model    : sleep(0.0001 * (i % 20)) per item")

    run_workload("REQUIRED  cost = 0.0001 * (i % 20)", cost_mod20)
    run_workload("BONUS     cost = 0.0002 * (i // 50)  [ramp / heavy tail]", cost_ramp)

    print("\nNote: time.sleep has OS timer granularity (~0.1-1+ ms on Windows), so measured")
    print("times exceed the modeled ideal; Efficiency = ideal / measured total.")


if __name__ == "__main__":
    main()
