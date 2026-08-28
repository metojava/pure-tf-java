package puretf.nn;

/**
 * Keras-style early stopping on a monitored metric. The trainer feeds each
 * epoch's metric via {@link #update(double)}; when no improvement of at
 * least {@code minDelta} has occurred for {@code patience} consecutive
 * epochs, training stops. Best-weights snapshots are managed by the caller
 * through {@link #improved()}.
 */
public final class EarlyStopping {

    public enum Monitor { LOSS, ACCURACY }

    private final int patience;
    private final double minDelta;
    private double best;
    private int bad = 0;

    /** Monitor kind this instance watches (read by the trainer). */
    final Monitor monitor;

    public EarlyStopping(Monitor monitor, int patience, double minDelta) {
        this.monitor = monitor;
        this.patience = patience;
        this.minDelta = minDelta;
        best = monitor == Monitor.LOSS ? Double.POSITIVE_INFINITY
                : Double.NEGATIVE_INFINITY;
    }

    /** Feeds one epoch's metric; returns true when training should stop. */
    public boolean update(double metric) {
        boolean improved = monitor == Monitor.LOSS
                ? metric < best - minDelta
                : metric > best + minDelta;
        if (improved) {
            best = metric;
            bad = 0;
        } else if (++bad >= patience) {
            return true;
        }
        return false;
    }

    /** True for exactly the call on which the monitored value improved. */
    public boolean improved() { return bad == 0; }
}
