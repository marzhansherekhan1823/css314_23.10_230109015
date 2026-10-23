"""Task 4 (CPU-only path): inter-process transfer overhead.
Path A: pickle over a multiprocessing Pipe (serialize -> kernel pipe -> deserialize).
Path B: zero-copy multiprocessing.shared_memory (child attaches, no serialization).
Run: python3 task4.py [seed]
"""
import sys, time, statistics
import numpy as np
from multiprocessing import Process, Pipe, shared_memory, resource_tracker

SIZES = [100_000, 1_000_000, 10_000_000]
REPS = {100_000: 30, 1_000_000: 15, 10_000_000: 7}
DTYPE = np.int64


def worker(conn):
    """Persistent consumer so process start-up is NOT counted as transfer cost."""
    while True:
        msg = conn.recv()
        if msg[0] == "quit":
            return
        if msg[0] == "pickle":                      # Path A: msg[1] is the unpickled array
            arr = msg[1]
            conn.send(int(arr[-1]))                 # touch last element, ack
        elif msg[0] == "shm":                       # Path B: attach by name, no copy
            _, name, n = msg
            shm = shared_memory.SharedMemory(name=name)
            if sys.platform != "win32":
                # POSIX only: stop the child's resource tracker from reporting bogus
                # "leaked" segments; the parent owns/unlinks it. (Windows has no tracker.)
                resource_tracker.unregister(shm._name, "shared_memory")
            arr = np.ndarray((n,), dtype=DTYPE, buffer=shm.buf)
            last = int(arr[-1])
            del arr
            shm.close()
            conn.send(last)


def med(f, reps):
    f()  # warm-up
    return statistics.median(f() for _ in range(reps))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    rng = np.random.default_rng(seed)
    parent, child = Pipe()
    p = Process(target=worker, args=(child,), daemon=True)
    p.start()

    print(f"seed={seed} dtype=int64 cpu_count={__import__('os').cpu_count()}")
    print("N | bytes(MB) | Path A pickle/pipe (ms) | Path B shm total (ms) | "
          "Path B attach-only (ms) | A/B ratio | Check")
    ok_all = True
    for n in SIZES:
        data = rng.integers(1, 11, size=n, dtype=DTYPE)
        expect = int(data[-1])

        def path_a():
            t = time.perf_counter()
            parent.send(("pickle", data))
            r = parent.recv()
            dt = (time.perf_counter() - t) * 1e3
            assert r == expect
            return dt

        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        view = np.ndarray((n,), dtype=DTYPE, buffer=shm.buf)

        def path_b_total():          # includes copying data into the shared segment
            t = time.perf_counter()
            view[:] = data
            parent.send(("shm", shm.name, n))
            r = parent.recv()
            dt = (time.perf_counter() - t) * 1e3
            assert r == expect
            return dt

        def path_b_attach():         # data already resident in shm (true zero-copy handoff)
            t = time.perf_counter()
            parent.send(("shm", shm.name, n))
            r = parent.recv()
            dt = (time.perf_counter() - t) * 1e3
            assert r == expect
            return dt

        a = med(path_a, REPS[n])
        b_tot = med(path_b_total, REPS[n])
        b_att = med(path_b_attach, REPS[n])
        print(f"{n:,} | {data.nbytes/1e6:.1f} | {a:.3f} | {b_tot:.3f} | {b_att:.3f} | "
              f"{a/b_tot:.1f}x (total) / {a/b_att:.0f}x (attach) | PASS (child received correct data)")
        del view
        shm.close(); shm.unlink()

    parent.send(("quit",))
    p.join(timeout=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
