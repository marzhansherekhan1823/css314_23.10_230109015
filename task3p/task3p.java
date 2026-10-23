// Task 3: Prefix scan - serial vs. 2-pass chunked parallel (reduce-then-scan)
// Run: java task3.java [seed] [threads]      (Java 11+ single-file mode)
import java.util.Arrays;
import java.util.Random;

class Task3 {
    static final int N = 2_000_000;

    static void serialScan(int[] in, long[] out) {
        out[0] = in[0];
        for (int i = 1; i < in.length; i++) out[i] = out[i - 1] + in[i];
    }

    static void forkJoin(int T, java.util.function.IntConsumer fn) throws InterruptedException {
        Thread[] th = new Thread[T - 1];
        for (int t = 1; t < T; t++) { final int id = t; th[t - 1] = new Thread(() -> fn.accept(id)); th[t - 1].start(); }
        fn.accept(0);
        for (Thread x : th) x.join();
    }

    static void parallelScan(int[] in, long[] out, int T) throws InterruptedException {
        final int n = in.length, chunk = (n + T - 1) / T;
        final long[] sums = new long[T], offs = new long[T];
        // Pass 1: local chunk sums
        forkJoin(T, t -> {
            int lo = Math.min(n, t * chunk), hi = Math.min(n, lo + chunk);
            long s = 0;
            for (int i = lo; i < hi; i++) s += in[i];
            sums[t] = s;
        });
        // Offset reduction (exclusive scan over T chunk sums)
        for (int t = 1; t < T; t++) offs[t] = offs[t - 1] + sums[t - 1];
        // Pass 2: local scan seeded with chunk offset
        forkJoin(T, t -> {
            int lo = Math.min(n, t * chunk), hi = Math.min(n, lo + chunk);
            long acc = offs[t];
            for (int i = lo; i < hi; i++) { acc += in[i]; out[i] = acc; }
        });
    }

    interface Job { void run() throws Exception; }

    static double medianMs(Job job, int reps) throws Exception {
        double[] v = new double[reps];
        for (int r = 0; r < reps; r++) {
            long a = System.nanoTime(); job.run(); long b = System.nanoTime();
            v[r] = (b - a) / 1e6;
        }
        Arrays.sort(v);
        return v[reps / 2];
    }

    public static void main(String[] args) throws Exception {
        long seed = args.length > 0 ? Long.parseLong(args[0]) : 42;
        int hw = Runtime.getRuntime().availableProcessors();
        int T = args.length > 1 ? Integer.parseInt(args[1]) : Math.max(4, hw);

        Random rng = new Random(seed);
        int[] in = new int[N];
        for (int i = 0; i < N; i++) in[i] = rng.nextInt(10) + 1;   // 1..10
        long[] s = new long[N], p = new long[N];

        for (int i = 0; i < 20; i++) { serialScan(in, s); parallelScan(in, p, T); }  // JIT warm-up
        double ts = medianMs(() -> serialScan(in, s), 25);
        double tp = medianMs(() -> parallelScan(in, p, T), 25);

        boolean lastOk = p[N - 1] == s[N - 1];
        boolean fullOk = Arrays.equals(p, s);
        boolean pass = lastOk && fullOk;

        System.out.printf("N=%d seed=%d threads=%d hw_cores=%d%n", N, seed, T, hw);
        System.out.println("Serial Time (ms) | Parallel Time (ms) | Speedup | Verification Status");
        System.out.printf("%.3f | %.3f | %.2fx | %s%n", ts, tp, ts / tp, pass ? "PASS" : "FAIL");
        System.out.printf("serial_last=%d parallel_last=%d%n", s[N - 1], p[N - 1]);
        if (!lastOk) { System.err.println("ASSERTION FAILED: parallel_output[-1] != serial_output[-1]"); System.exit(1); }
        if (!fullOk) { System.err.println("FULL-ARRAY MISMATCH"); System.exit(1); }
    }
}
