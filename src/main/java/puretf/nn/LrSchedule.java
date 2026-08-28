package puretf.nn;

/**
 * Learning-rate schedule: returns the base LR for an epoch.
 * Static factories cover the common shapes; lambdas give full control.
 */
@FunctionalInterface
public interface LrSchedule {

    /** LR used for {@code epoch} (0-based). */
    float lrAt(int epoch);

    /** Constant schedule (plain SGD/Adam behavior). */
    static LrSchedule constant(float lr) {
        return epoch -> lr;
    }

    /** Halve the LR every {@code dropEvery} epochs. */
    static LrSchedule step(float lr, int dropEvery) {
        return epoch -> (float) (lr * Math.pow(0.5, epoch / (double) dropEvery));
    }

    /** Cosine annealing from {@code maxLr} down to {@code minLr}. */
    static LrSchedule cosine(float maxLr, float minLr, int totalEpochs) {
        return epoch -> {
            if (totalEpochs <= 1) return maxLr;
            double t = Math.PI * epoch / (totalEpochs - 1);
            return (float) (minLr + 0.5 * (maxLr - minLr) * (1 + Math.cos(t)));
        };
    }

    /** Exponential decay: lr * decay^epoch. */
    static LrSchedule exponential(float lr, float decay) {
        return epoch -> (float) (lr * Math.pow(decay, epoch));
    }
}
