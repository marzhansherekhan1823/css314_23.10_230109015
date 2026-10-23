import java.util.Arrays;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Task 8 - Parallel Tile Reduction (CPU path): 1024 x 1024 matrix multiplication C = A x B
 *
 * Run (JDK 11+):   java Task8.java
 *
 * Variants (each run single-threaded and on all logical cores):
 *   1. Non-tiled  i-j-k : textbook triple loop. B is read down a column (stride = 8 KB),
 *                          so almost every B access touches a new cache line -> cache-miss heavy.
 *   2. Non-tiled  i-k-j : same maths, loop order changed so B and C are read along rows (bonus
 *                          reference: shows how much of the win is just access order).
 *   3. Tiled 16x16      : 3 tiles = 6 KB working set   -> fits L1 data cache
 *   4. Tiled 64x64      : 3 tiles = 96 KB working set  -> spills L1, fits L2
 *
 * Parallelism: dynamic task queue (AtomicInteger). Non-tiled tasks = bands of 4 rows of C;
 * tiled tasks = one (ii, jj) tile of C, so each C element is written by exactly one thread.
 *
 * Cache behaviour: hardware miss counters are not readable from plain Java, so memory-stall cost
 * is ESTIMATED from runtime: (T_variant - T_best) / n^3 = extra nanoseconds per multiply-add
 * attributable to cache misses and poor locality, relative to the fastest single-thread variant.
 */
public class Task8 {
    static final String STUDENT_ID = "230109015";
    static final int N = 1024;
    static final int ROWS_PER_TASK = 4;
    static final int REPS = 3;

    record Variant(String name, String type, int bs) {
        long workingSetBytes(int n) {
            return switch (type) {
                case "tile" -> 3L * bs * bs * 8;
                default -> 8L * n * n;           // streams through whole B matrix
            };
        }
        int tasks(int n) {
            return type.equals("tile") ? (n / bs) * (n / bs) : n / ROWS_PER_TASK;
        }
    }

    static void runTask(Variant v, int t, double[] a, double[] b, double[] c, int n) {
        switch (v.type()) {
            case "ijk" -> {
                int lo = t * ROWS_PER_TASK, hi = lo + ROWS_PER_TASK;
                for (int i = lo; i < hi; i++) {
                    for (int j = 0; j < n; j++) {
                        double sum = 0;
                        for (int k = 0; k < n; k++) sum += a[i * n + k] * b[k * n + j];
                        c[i * n + j] = sum;
                    }
                }
            }
            case "ikj" -> {
                int lo = t * ROWS_PER_TASK, hi = lo + ROWS_PER_TASK;
                for (int i = lo; i < hi; i++) {
                    for (int k = 0; k < n; k++) {
                        double aik = a[i * n + k];
                        for (int j = 0; j < n; j++) c[i * n + j] += aik * b[k * n + j];
                    }
                }
            }
            case "tile" -> {
                int bs = v.bs();
                int perRow = n / bs;
                int ii = (t / perRow) * bs, jj = (t % perRow) * bs;
                for (int kk = 0; kk < n; kk += bs) {
                    for (int i = ii; i < ii + bs; i++) {
                        for (int k = kk; k < kk + bs; k++) {
                            double aik = a[i * n + k];
                            for (int j = jj; j < jj + bs; j++) c[i * n + j] += aik * b[k * n + j];
                        }
                    }
                }
            }
            default -> throw new IllegalStateException();
        }
    }

    /** Zeroes C, runs the variant with `threads` workers pulling tasks dynamically; returns ms. */
    static double timedRun(Variant v, int threads, double[] a, double[] b, double[] c, int n) throws Exception {
        Arrays.fill(c, 0.0);
        AtomicInteger next = new AtomicInteger();
        int total = v.tasks(n);
        Thread[] ts = new Thread[threads];
        long t0 = System.nanoTime();
        for (int w = 0; w < threads; w++) {
            ts[w] = new Thread(() -> {
                int t;
                while ((t = next.getAndIncrement()) < total) runTask(v, t, a, b, c, n);
            });
            ts[w].start();
        }
        for (Thread t : ts) t.join();
        return (System.nanoTime() - t0) / 1e6;
    }

    static double median(double[] v) {
        double[] c = v.clone();
        Arrays.sort(c);
        return c[c.length / 2];
    }

    static String bytes(long b) {
        return b >= (1 << 20) ? String.format("%d MB", b >> 20) : String.format("%d KB", b >> 10);
    }

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.ROOT);
        int cores = Runtime.getRuntime().availableProcessors();

        System.out.println("Student ID    : " + STUDENT_ID);
        System.out.println("OS            : " + System.getProperty("os.name") + " " + System.getProperty("os.version")
                + " (" + System.getProperty("os.arch") + ")");
        System.out.println("Java          : " + System.getProperty("java.version"));
        System.out.println("Logical cores : " + cores);
        System.out.println("Matrix size   : " + N + " x " + N + " float64  (" + bytes(8L * N * N) + " per matrix)");
        System.out.println("Work          : 2*n^3 = " + String.format("%.3f", 2.0 * N * N * N / 1e9) + " GFLOP per multiply");
        System.out.println("Repeats       : median of " + REPS + " runs per cell");
        System.out.println("GPU path      : skipped (CPU parallel path chosen)");
        System.out.println();

        Variant[] variants = {
            new Variant("Non-tiled i-j-k", "ijk", 0),
            new Variant("Non-tiled i-k-j (bonus)", "ikj", 0),
            new Variant("Tiled 16x16", "tile", 16),
            new Variant("Tiled 64x64", "tile", 64),
        };

        // Inputs (seeded from Student ID)
        Random rnd = new Random(Long.parseLong(STUDENT_ID));
        double[] a = new double[N * N], b = new double[N * N], c = new double[N * N];
        for (int i = 0; i < N * N; i++) { a[i] = rnd.nextDouble(); b[i] = rnd.nextDouble(); }

        // JIT warm-up on a small problem
        int wn = 256;
        double[] wa = Arrays.copyOf(a, wn * wn), wb = Arrays.copyOf(b, wn * wn), wc = new double[wn * wn];
        for (int rep = 0; rep < 3; rep++)
            for (Variant v : variants) { timedRun(v, 1, wa, wb, wc, wn); timedRun(v, cores, wa, wb, wc, wn); }

        double[] t1 = new double[variants.length], tn = new double[variants.length];
        double[] reference = null;
        double maxErr = 0;

        for (int vi = 0; vi < variants.length; vi++) {
            Variant v = variants[vi];
            double[] s = new double[REPS], p = new double[REPS];
            for (int r = 0; r < REPS; r++) s[r] = timedRun(v, 1, a, b, c, N);
            if (reference == null) reference = c.clone();
            for (int i = 0; i < c.length; i++) maxErr = Math.max(maxErr, Math.abs(c[i] - reference[i]));
            for (int r = 0; r < REPS; r++) p[r] = timedRun(v, cores, a, b, c, N);
            for (int i = 0; i < c.length; i++) maxErr = Math.max(maxErr, Math.abs(c[i] - reference[i]));
            t1[vi] = median(s);
            tn[vi] = median(p);
        }

        double gflop = 2.0 * N * N * N / 1e9;
        double base = t1[0];
        double best1 = Arrays.stream(t1).min().getAsDouble();
        double ops = (double) N * N * N;

        String line = "-".repeat(118);
        System.out.println(line);
        System.out.printf("%-24s | %9s | %10s | %8s | %10s | %8s | %8s | %11s | %s%n",
                "Variant", "Work set", "1 thr (ms)", "GFLOP/s", cores + " thr (ms)", "GFLOP/s",
                "MT speedup", "1T vs naive", "Est. stall ns/FMA (1T)");
        System.out.println(line);
        for (int vi = 0; vi < variants.length; vi++) {
            Variant v = variants[vi];
            double stall = (t1[vi] - best1) * 1e6 / ops;     // ms -> ns, per multiply-add
            System.out.printf("%-24s | %9s | %10.1f | %8.2f | %10.1f | %8.2f | %9.2fx | %10.2fx | %.3f%n",
                    v.name(), bytes(v.workingSetBytes(N)), t1[vi], gflop / (t1[vi] / 1000.0),
                    tn[vi], gflop / (tn[vi] / 1000.0), t1[vi] / tn[vi], base / t1[vi], stall);
        }
        System.out.println(line);

        int bestIdx = 0;
        for (int vi = 1; vi < variants.length; vi++) if (t1[vi] < t1[bestIdx]) bestIdx = vi;
        int bestMt = 0;
        for (int vi = 1; vi < variants.length; vi++) if (tn[vi] < tn[bestMt]) bestMt = vi;

        System.out.println();
        System.out.println("ANALYSIS");
        System.out.printf("Correctness           : max |C - C_reference| over all runs = %.3e %s%n",
                maxErr, maxErr < 1e-8 ? "(PASS)" : "(CHECK)");
        System.out.printf("Fastest single-thread : %s (%.1f ms)%n", variants[bestIdx].name(), t1[bestIdx]);
        System.out.printf("Fastest multi-thread  : %s (%.1f ms on %d threads)%n", variants[bestMt].name(), tn[bestMt], cores);
        int bestTile = t1[2] <= t1[3] ? 2 : 3;
        System.out.printf("Tiling benefit        : naive i-j-k is %.2fx slower than %s single-threaded%n",
                t1[0] / t1[bestTile], variants[bestTile].name());
        System.out.printf("Loop order alone      : i-k-j (no tiling) is %.2fx %s than %s single-threaded%n",
                Math.max(t1[1], t1[bestTile]) / Math.min(t1[1], t1[bestTile]),
                t1[1] <= t1[bestTile] ? "faster" : "slower", variants[bestTile].name());
        System.out.printf("Tile size effect      : 16x16 = %.1f ms vs 64x64 = %.1f ms (single-thread), 16x16 fits L1 (6 KB), 64x64 needs L2 (96 KB)%n",
                t1[2], t1[3]);
        System.out.println("Cache-miss note       : the naive i-j-k loop strides 8 KB through B on every inner step, so each B access");
        System.out.println("                        lands on a different cache line (and page). The stall column is an ESTIMATE from");
        System.out.println("                        runtime differences, not a hardware miss counter.");
    }
}
