import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Task 2 - Pseudo-Random Generator State Bottleneck (Monte Carlo pi, 40,000,000 samples)
 *
 * Run (JDK 11+):   java Task2.java
 *
 * Both runs use 4 threads x 10,000,000 samples and the SAME generator algorithm (java.util.Random),
 * so the only difference is whether the generator state is shared.
 *
 *   1. SHARED      : one Random instance, every (x, y) draw happens inside a ReentrantLock critical
 *                    section -> the 4 threads serialize on the generator state.
 *   2. INDEPENDENT : each thread owns its own Random seeded with (Student_ID * 10) + thread_index
 *                    -> no lock, no shared state, fully parallel.
 *
 * The shared generator is seeded with Student_ID * 10 (thread_index 0 convention).
 */
public class Task2 {
    static final long STUDENT_ID = 230109015L;
    static final int THREADS = 4;
    static final long TOTAL = 40_000_000L;
    static final long PER_THREAD = TOTAL / THREADS;
    static final int TRIALS = 3;

    record Result(double pi, double ms) { }

    static Result runShared(long perThread) throws Exception {
        Random shared = new Random(STUDENT_ID * 10);
        ReentrantLock lock = new ReentrantLock();
        long[] hits = new long[THREADS];
        return execute(perThread, hits, idx -> () -> {
            long h = 0;
            for (long k = 0; k < perThread; k++) {
                double x, y;
                lock.lock();
                try {
                    x = shared.nextDouble();
                    y = shared.nextDouble();
                } finally {
                    lock.unlock();
                }
                if (x * x + y * y <= 1.0) h++;
            }
            hits[idx] = h;
        });
    }

    static Result runIndependent(long perThread) throws Exception {
        long[] hits = new long[THREADS * 8];             // 64-byte spacing: no false sharing on results
        return executeStrided(perThread, hits, idx -> () -> {
            Random rng = new Random(STUDENT_ID * 10 + idx);
            long h = 0;
            for (long k = 0; k < perThread; k++) {
                double x = rng.nextDouble();
                double y = rng.nextDouble();
                if (x * x + y * y <= 1.0) h++;
            }
            hits[idx * 8] = h;
        });
    }

    interface Factory { Runnable make(int idx); }

    static Result execute(long perThread, long[] hits, Factory f) throws Exception {
        return launch(perThread, hits, 1, f);
    }

    static Result executeStrided(long perThread, long[] hits, Factory f) throws Exception {
        return launch(perThread, hits, 8, f);
    }

    static Result launch(long perThread, long[] hits, int stride, Factory f) throws Exception {
        CountDownLatch ready = new CountDownLatch(THREADS), go = new CountDownLatch(1);
        Thread[] ts = new Thread[THREADS];
        for (int i = 0; i < THREADS; i++) {
            Runnable body = f.make(i);
            ts[i] = new Thread(() -> {
                ready.countDown();
                try { go.await(); } catch (InterruptedException e) { return; }
                body.run();
            });
            ts[i].start();
        }
        ready.await();
        long t0 = System.nanoTime();
        go.countDown();
        for (Thread t : ts) t.join();
        double ms = (System.nanoTime() - t0) / 1e6;
        long total = 0;
        for (int i = 0; i < THREADS; i++) total += hits[i * stride];
        return new Result(4.0 * total / (perThread * THREADS), ms);
    }

    static Result median(Result[] r) {
        Result[] c = r.clone();
        Arrays.sort(c, (a, b) -> Double.compare(a.ms(), b.ms()));
        return c[c.length / 2];
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);

        System.out.println("Student ID       : " + STUDENT_ID);
        System.out.println("OS               : " + System.getProperty("os.name") + " "
                + System.getProperty("os.version") + " (" + System.getProperty("os.arch") + ")");
        System.out.println("Java             : " + System.getProperty("java.version"));
        System.out.println("Logical cores    : " + Runtime.getRuntime().availableProcessors());
        System.out.println("Threads          : " + THREADS + " x " + String.format("%,d", PER_THREAD)
                + " samples = " + String.format("%,d", TOTAL));
        System.out.println("Seeds            : shared = " + (STUDENT_ID * 10) + " ; independent = "
                + (STUDENT_ID * 10) + " + thread_index (0.." + (THREADS - 1) + ")");
        System.out.println("Trials           : " + TRIALS + " per variant (median wall time reported)");
        System.out.println();

        // JIT warm-up
        runShared(200_000);
        runIndependent(200_000);

        Result[] shared = new Result[TRIALS], indep = new Result[TRIALS];
        for (int t = 0; t < TRIALS; t++) {
            shared[t] = runShared(PER_THREAD);
            indep[t] = runIndependent(PER_THREAD);
        }
        Result s = median(shared), i = median(indep);

        String line = "-".repeat(78);
        System.out.println(line);
        System.out.printf("%-22s | %-12s | %-14s | %s%n", "Variant", "Computed pi", "Abs. error", "Wall time");
        System.out.println(line);
        System.out.printf("%-22s | %-12.8f | %-14.3e | %.2f ms%n", "Shared PRNG + lock", s.pi(), Math.abs(s.pi() - Math.PI), s.ms());
        System.out.printf("%-22s | %-12.8f | %-14.3e | %.2f ms%n", "Independent PRNGs", i.pi(), Math.abs(i.pi() - Math.PI), i.ms());
        System.out.println(line);
        System.out.printf("Reference Math.PI      : %.8f%n", Math.PI);
        System.out.printf("Speedup from removing generator-state locking : %.2fx  (%.2f ms -> %.2f ms)%n",
                s.ms() / i.ms(), s.ms(), i.ms());
        System.out.println();
        System.out.print("Shared trials (ms)      : ");
        for (Result r : shared) System.out.printf("%.1f  ", r.ms());
        System.out.println();
        System.out.print("Independent trials (ms) : ");
        for (Result r : indep) System.out.printf("%.1f  ", r.ms());
        System.out.println();
        System.out.println("Note: the independent-seed result is deterministic (same pi every run); the shared-generator");
        System.out.println("pi varies slightly between runs because thread interleaving changes who draws which numbers.");
    }
}
