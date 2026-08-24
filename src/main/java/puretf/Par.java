package puretf;

import java.util.function.BiConsumer;

/**
 * Work-splitting helper for the heavy inner loops of Tensor ops. The
 * autograd tape stays strictly sequential; only per-op loops are split
 * across the common ForkJoin pool, partitioned so every worker writes
 * DISJOINT output slices (deterministic, no locks in the hot path).
 */
final class Par {

    static final int THREADS = Runtime.getRuntime().availableProcessors();

    private Par() {}

    /** True when the work amount justifies pool dispatch overhead. */
    static boolean worth(long ops) {
        return THREADS > 1 && ops >= 1_000_000;
    }

    /**
     * Runs {@code body(workerId, threadCount)} on all pool threads; each
     * worker must only touch its own disjoint slice of the output.
     */
    static void run(BiConsumer<Integer, Integer> body) {
        if (THREADS == 1) {
            body.accept(0, 1);
            return;
        }
        java.util.stream.IntStream.range(0, THREADS).parallel()
                .forEach(t -> body.accept(t, THREADS));
    }

    /** Range of indices [n, total) handled by worker {@code id} of {@code T}. */
    static int start(int id, int total, int T) {
        return (int) ((long) total * id / T);
    }
}
