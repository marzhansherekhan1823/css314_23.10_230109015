import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Task 6 - Synchronization Cost & Data Loss
 *
 * Run (JDK 11+):   java Task6.java
 *
 * 4 threads, each increments a shared counter 2,000,000 times. Expected total = 8,000,000.
 *   1. UNSAFE  : plain `counter++` on a shared int (read-modify-write race -> lost updates)
 *   2. LOCK    : ReentrantLock (mutex) around every increment -> correct, slower
 *   3. ATOMIC  : AtomicInteger.incrementAndGet() (hardware atomic LOCK XADD) -> correct
 *
 * The unsafe result is non-deterministic, so it is run several times. The unsafe counter is
 * read and written with opaque (non-atomic, no fence, no lock) accesses so that the JIT cannot
 * collapse the loop into a single addition; every iteration performs a real load -> add -> store
 * on shared memory, which is exactly the read-modify-write sequence that races between cores.
 */
public class Task6 {
    static final String STUDENT_ID = "230109015";
    static final int THREADS = 4;
    static final int PER_THREAD = 2_000_000;
    static final int EXPECTED = THREADS * PER_THREAD;
    static final int TRIALS = 5;

    static final int[] UNSAFE = new int[1];
    static final VarHandle VH = MethodHandles.arrayElementVarHandle(int[].class);
    static int lockedCounter;
    static final ReentrantLock LOCK = new ReentrantLock();
    static final AtomicInteger ATOMIC = new AtomicInteger();

    interface Body { void run(int iters); }

    static void unsafeBody(int iters) { for (int i = 0; i < iters; i++) {
            int v = (int) VH.getOpaque(UNSAFE, 0);   // load
            VH.setOpaque(UNSAFE, 0, v + 1);          // add + store (not atomic with the load)
        }
    }

    static void lockBody(int iters) {
        for (int i = 0; i < iters; i++) {
            LOCK.lock();
            try { lockedCounter++; } finally { LOCK.unlock(); }
        }
    }

    static void atomicBody(int iters) { for (int i = 0; i < iters; i++) ATOMIC.incrementAndGet(); }

    /** Runs THREADS threads released simultaneously; returns wall-clock ms. */
    static double run(Body body, int iters) throws Exception {
        CountDownLatch ready = new CountDownLatch(THREADS);
        CountDownLatch go = new CountDownLatch(1);
        Thread[] ts = new Thread[THREADS];
        for (int i = 0; i < THREADS; i++) {
            ts[i] = new Thread(() -> {
                ready.countDown();
                try { go.await(); } catch (InterruptedException e) { return; }
                body.run(iters);
            });
            ts[i].start();
        }
        ready.await();
        long t0 = System.nanoTime();
        go.countDown();
        for (Thread t : ts) t.join();
        return (System.nanoTime() - t0) / 1e6;
    }

    static double median(double[] v) {
        double[] c = v.clone();
        Arrays.sort(c);
        return c[c.length / 2];
    }

    public static void main(String[] args) throws Exception {
        System.out.println("Student ID       : " + STUDENT_ID);
        System.out.println("OS               : " + System.getProperty("os.name") + " "
                + System.getProperty("os.version") + " (" + System.getProperty("os.arch") + ")");
        System.out.println("Java             : " + System.getProperty("java.version"));
        System.out.println("Logical cores    : " + Runtime.getRuntime().availableProcessors());
        System.out.println("Threads          : " + THREADS + " x " + String.format("%,d", PER_THREAD) + " increments");
        System.out.println("Expected final   : " + String.format("%,d", EXPECTED));
        System.out.println();

        // Warm-up (JIT) for all three bodies
        run(Task6::unsafeBody, 200_000);
        run(Task6::lockBody, 200_000);
        run(Task6::atomicBody, 200_000);

        double[] tUnsafe = new double[TRIALS], tLock = new double[TRIALS], tAtomic = new double[TRIALS];
        int[] unsafeResults = new int[TRIALS];
        boolean lockOk = true, atomicOk = true;

        for (int t = 0; t < TRIALS; t++) {
            UNSAFE[0] = 0;
            tUnsafe[t] = run(Task6::unsafeBody, PER_THREAD);
            unsafeResults[t] = UNSAFE[0];

            lockedCounter = 0;
            tLock[t] = run(Task6::lockBody, PER_THREAD);
            lockOk &= lockedCounter == EXPECTED;

            ATOMIC.set(0);
            tAtomic[t] = run(Task6::atomicBody, PER_THREAD);
            atomicOk &= ATOMIC.get() == EXPECTED;
        }

        System.out.println("PART 1 - UNSYNCHRONIZED (race condition), " + TRIALS + " trials");
        String l1 = "-".repeat(66);
        System.out.println(l1);
        System.out.printf("%5s | %12s | %12s | %12s | %10s%n", "Trial", "Expected", "Actual", "Lost", "Lost %");
        System.out.println(l1);
        long totalLost = 0;
        int worst = 0;
        for (int t = 0; t < TRIALS; t++) {
            int lost = EXPECTED - unsafeResults[t];
            totalLost += lost;
            worst = Math.max(worst, lost);
            System.out.printf("%5d | %,12d | %,12d | %,12d | %9.4f%%%n",
                    t + 1, EXPECTED, unsafeResults[t], lost, 100.0 * lost / EXPECTED);
        }
        System.out.println(l1);
        double avgLost = (double) totalLost / TRIALS;
        System.out.printf("Average actual value : %,.0f   |   Average lost updates : %,.0f (%.4f%%)%n",
                EXPECTED - avgLost, avgLost, 100.0 * avgLost / EXPECTED);
        System.out.printf("Worst trial          : %,d updates lost (%.4f%%)%n", worst, 100.0 * worst / EXPECTED);

        System.out.println();
        System.out.println("PART 2 - SYNCHRONIZED VERSIONS (median wall-clock of " + TRIALS + " trials)");
        String l2 = "-".repeat(66);
        System.out.println(l2);
        System.out.printf("%-26s | %10s | %9s | %s%n", "Variant", "Time (ms)", "vs unsafe", "Final value");
        System.out.println(l2);
        double mU = median(tUnsafe), mL = median(tLock), mA = median(tAtomic);
        System.out.printf("%-26s | %10.2f | %8.2fx | %s%n", "Unsafe (no sync)", mU, 1.0, "WRONG (see above)");
        System.out.printf("%-26s | %10.2f | %8.2fx | %,d %s%n", "ReentrantLock (mutex)", mL, mL / mU, EXPECTED, lockOk ? "(correct)" : "(INCORRECT)");
        System.out.printf("%-26s | %10.2f | %8.2fx | %,d %s%n", "AtomicInteger (HW atomic)", mA, mA / mU, EXPECTED, atomicOk ? "(correct)" : "(INCORRECT)");
        System.out.println(l2);

        System.out.printf("%nSynchronization penalty (lock vs unsafe)   : +%.2f ms  (%.2fx slower)%n", mL - mU, mL / mU);
        System.out.printf("Synchronization penalty (atomic vs unsafe) : +%.2f ms  (%.2fx slower)%n", mA - mU, mA / mU);
        System.out.printf("Atomic vs lock                             : atomic is %.2fx %s than lock%n",
                mL >= mA ? mL / mA : mA / mL, mL >= mA ? "faster" : "slower");
    }
}
