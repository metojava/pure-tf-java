package puretf.data;

import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Random;

/**
 * Generic numeric dataset helper: wraps float feature rows and (optionally)
 * int class labels, and provides the common manipulations an ML workflow
 * needs before handing data to an engine:
 *
 * <ul>
 *   <li>{@link #shuffle} - stable, seeded permutation of the samples</li>
 *   <li>{@link #split} - deterministic train / validation split (two views)</li>
 *   <li>{@link #batches} - mini-batch iteration over rows</li>
 *   <li>{@link #subsample} - take an evenly-spread slice (fast prototyping)</li>
 * </ul>
 *
 * Features and labels stay aligned under every operation because only an
 * internal permutation and a visible length are ever mutated; the underlying
 * feature/label arrays are never copied or reordered.
 */
public final class Dataset {

    private final float[][] X;
    private final int[] Y;         // may be null for unlabeled data
    private final int[] order;     // current permutation of sample indices
    private int len;               // number of visible (leading) samples

    /** Wrap features and (optionally) integer class labels. */
    public Dataset(float[][] X, int[] Y) {
        if (Y != null && Y.length != X.length)
            throw new IllegalArgumentException("X and Y must have equal length");
        this.X = X;
        this.Y = Y;
        this.order = new int[X.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        this.len = order.length;
        shuffle(order, len, new Random(7L));
    }

    public int size() { return len; }
    public int featureDim() { return X[0].length; }
    public boolean labeled() { return Y != null; }

    /** Row-major feature vector of the i-th (permuted) sample. */
    public float[] sample(int i) { return X[order[i]]; }

    /** Class label of the i-th (permuted) sample, or -1 if unlabeled. */
    public int label(int i) { return Y == null ? -1 : Y[order[i]]; }

    /** All features in the current permutation: float[n][d]. */
    public float[][] x() {
        float[][] out = new float[len][];
        for (int i = 0; i < len; i++) out[i] = X[order[i]];
        return out;
    }

    /** All labels in the current permutation: int[n]. Null if unlabeled. */
    public int[] y() { return Y == null ? null : labelsOf(order, 0, len); }

    /** Stable, seeded shuffle of the current ordering. Returns this. */
    public Dataset shuffle(long seed) {
        shuffle(order, len, new Random(seed));
        return this;
    }

    /**
     * Deterministic split into [trainFrac, 1-trainFrac]. Returns two views
     * over the same data: the left side used for training, the right (until
     * the visible length) reserved for validation.
     */
    public Split split(double trainFrac) {
        int n = (int) Math.round(len * trainFrac);
        n = Math.max(0, Math.min(len, n));
        return new Split(view(0, n), view(n, len));
    }

    /**
     * First {@code k} evenly-spread samples (including the last one), an easy
     * way to run quick experiments on a fraction of a large dataset.
     */
    public Dataset subsample(int k) {
        k = Math.min(Math.max(k, 0), len);
        int[] keep = new int[k];
        if (k == len) {
            System.arraycopy(order, 0, keep, 0, k);
        } else if (k > 0) {
            for (int i = 0; i < k; i++)
                keep[i] = order[(int) Math.round(i * (len - 1) / (double) (k - 1))];
        }
        Dataset sub = new Dataset(X, Y);
        System.arraycopy(keep, 0, sub.order, 0, keep.length);
        sub.len = keep.length;
        return sub;
    }

    /**
     * Iterate the dataset in mini-batches of up to {@code batchSize}. Each
     * batch exposes aligned features/labels as fresh row-major arrays.
     */
    public Iterable<Batch> batches(final int batchSize) {
        return () -> new Iterator<>() {
            private int start = 0;
            public boolean hasNext() { return start < len; }
            public Batch next() {
                if (start >= len) throw new NoSuchElementException();
                int size = Math.min(batchSize, len - start);
                Batch b = new Batch(X, Y, order, start, size);
                start += size;
                return b;
            }
        };
    }

    // ------------------------- private helpers ----------------------------

    /** A read-only window view sharing this dataset's arrays. */
    private Dataset view(int from, int to) {
        Dataset v = new Dataset(X, Y);
        System.arraycopy(order, from, v.order, 0, to - from);
        v.len = to - from;
        return v;
    }

    private static void shuffle(int[] arr, int n, Random rnd) {
        for (int i = n - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
        }
    }

    private static int[] labelsOf(int[] order, int from, int to) {
        int[] out = new int[to - from];
        System.arraycopy(order, from, out, 0, to - from);
        return out;
    }

    // --------------------------- result types -----------------------------

    /** Result of {@link #split}: mutually exclusive train and val views. */
    public record Split(Dataset train, Dataset val) {}

    /** One mini-batch: aligned slice of the current permutation. */
    public static final class Batch {
        private final float[][] X;
        private final int[] Y;
        private final int[] order;
        private final int from, size;

        private Batch(float[][] X, int[] Y, int[] order, int from, int size) {
            this.X = X; this.Y = Y; this.order = order; this.from = from; this.size = size;
        }

        public int size() { return size; }

        /** Features of this batch: float[size][d]. */
        public float[][] x() {
            float[][] out = new float[size][];
            for (int i = 0; i < size; i++) out[i] = X[order[from + i]];
            return out;
        }

        /** Labels of this batch: int[size], or null if unlabeled. */
        public int[] y() {
            if (Y == null) return null;
            int[] out = new int[size];
            for (int i = 0; i < size; i++) out[i] = Y[order[from + i]];
            return out;
        }
    }
}
