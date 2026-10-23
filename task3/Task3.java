import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.concurrent.CountDownLatch;

/**
 * Task 3 - Cache Line Contention (False Sharing)
 *
 * Run (JDK 11+):   java Task3.java            (threads = detected physical cores)
 *                  java Task3.java 14         (override thread count)
 *
 * Three variants, N threads each, every thread performs 50,000,000 increments:
 *   1. UNPADDED SHARED : thread i increments counts[i]; neighbours share a 64-byte cache line
 *   2. PADDED SHARED   : thread i increments counts[i * 8]; each counter owns its own 64-byte line
 *   3. LOCAL VARIABLE  : thread i increments a local variable, writes counts[i] once at the end
 *
 * Variants 1 and 2 use opaque (non-volatile, no fence) loads/stores through a VarHandle so the
 * JIT cannot keep the counter in a register: every iteration really touches memory, which is
 * what exposes cache-line invalidation between cores.
 */
public class Task3 {
    static final String STUDENT_ID = "230109015";
    static final long ITERATIONS = 50_000_000L;
    static final int PAD = 8;                       // 8 longs = 64 bytes
    static final VarHandle VH = MethodHandles.arrayElementVarHandle(long[].class);

    interface Worker { void run(long[] arr, int idx, long iters); }

    // 1 & 2: memory increment on arr[idx] each iteration
    static void memoryIncrement(long[] arr, int idx, long iters) {
        for (long k = 0; k < iters; k++) {
            VH.setOpaque(arr, idx, (long) VH.getOpaque(arr, idx) + 1L);
        }
    }

    // 3: local accumulator, single write at the end
    static void localIncrement(long[] arr, int idx, long iters) {
        long local = 0;
        for (long k = 0; k < iters; k++) {
            local++;
        }
        arr[idx] = local;
    }

    /** Runs n threads and returns wall-clock time in ms (from simultaneous release to last join). */
    static double run(int n, int stride, long iters, Worker w, long[] arr) throws Exception {
        CountDownLatch ready = new CountDownLatch(n);
        CountDownLatch go = new CountDownLatch(1);
        Thread[] ts = new Thread[n];
        for (int i = 0; i < n; i++) {
            final int idx = i * stride;
            ts[i] = new Thread(() -> {
                ready.countDown();
                try { go.await(); } catch (InterruptedException e) { return; }
                w.run(arr, idx, iters);
            });
            ts[i].start();
        }
        ready.await();
        long t0 = System.nanoTime();
        go.countDown();
        for (Thread t : ts) t.join();
        return (System.nanoTime() - t0) / 1e6;
    }

    static int physicalCores(int logical) {
        try {
            String os = System.getProperty("os.name").toLowerCase();
            Process p = null;
            if (os.contains("win")) {
                p = new ProcessBuilder("powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum")
                    .redirectErrorStream(true).start();
            } else if (os.contains("linux")) {
                p = new ProcessBuilder("sh", "-c",
                    "lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l")
                    .redirectErrorStream(true).start();
            } else if (os.contains("mac")) {
                p = new ProcessBuilder("sysctl", "-n", "hw.physicalcpu")
                    .redirectErrorStream(true).start();
            }
            if (p != null) {
                String out = new String(p.getInputStream().readAllBytes()).trim();
                int v = Integer.parseInt(out.split("\\s+")[0]);
                if (v > 0 && v <= logical) return v;
            }
        } catch (Exception ignored) { }
        return Math.max(1, logical / 2);            // fallback assumption: 2-way SMT
    }

    public static void main(String[] args) throws Exception {
        int logical = Runtime.getRuntime().availableProcessors();
        int n = args.length > 0 ? Integer.parseInt(args[0]) : physicalCores(logical);

        System.out.println("Student ID        : " + STUDENT_ID);
        System.out.println("OS                : " + System.getProperty("os.name") + " "
                + System.getProperty("os.version") + " (" + System.getProperty("os.arch") + ")");
        System.out.println("Java              : " + System.getProperty("java.version"));
        System.out.println("Logical cores     : " + logical);
        System.out.println("Threads used (N)  : " + n + (args.length > 0 ? " (override)" : " (physical cores)"));
        System.out.println("Increments/thread : " + String.format("%,d", ITERATIONS));
        System.out.println();

        // Warm-up so the JIT compiles all three loops before timing
        long[] warm = new long[n * PAD];
        run(n, 1, 2_000_000L, Task3::memoryIncrement, warm);
        run(n, PAD, 2_000_000L, Task3::memoryIncrement, warm);
        run(n, 1, 2_000_000L, Task3::localIncrement, warm);

        long[] unpadded = new long[n];
        double tUnpadded = run(n, 1, ITERATIONS, Task3::memoryIncrement, unpadded);

        long[] padded = new long[n * PAD];
        double tPadded = run(n, PAD, ITERATIONS, Task3::memoryIncrement, padded);

        long[] local = new long[n];
        double tLocal = run(n, 1, ITERATIONS, Task3::localIncrement, local);

        // Correctness check: every counter must equal ITERATIONS
        boolean ok = true;
        for (int i = 0; i < n; i++) {
            ok &= unpadded[i] == ITERATIONS && padded[i * PAD] == ITERATIONS && local[i] == ITERATIONS;
        }

        String line = "-".repeat(56);
        System.out.println(line);
        System.out.printf("%-30s | %12s | %s%n", "Variant", "Wall Time", "vs Unpadded");
        System.out.println(line);
        System.out.printf("%-30s | %9.2f ms | %8.2fx%n", "Unpadded shared (false share)", tUnpadded, 1.0);
        System.out.printf("%-30s | %9.2f ms | %8.2fx%n", "Padded shared (64B stride)", tPadded, tUnpadded / tPadded);
        System.out.printf("%-30s | %9.2f ms | %8.2fx%n", "Local variable, write once", tLocal, tUnpadded / tLocal);
        System.out.println(line);
        System.out.printf("Execution ratio Unpadded / Padded : %.2fx%n", tUnpadded / tPadded);
        System.out.printf("Execution ratio Unpadded / Local  : %.2fx%n", tUnpadded / tLocal);
        System.out.println("Counter check (all == 50,000,000) : " + (ok ? "PASS" : "FAIL"));
    }
}
