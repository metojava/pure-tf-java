package puretf.nn;

/** Pooling/conv padding mode, mirroring TF semantics. */
public enum Padding {
    VALID(0), SAME(1);

    public final int code;

    Padding(int code) { this.code = code; }

    public static Padding of(int code) {
        return code == 0 ? VALID : SAME;
    }

    /** Output size along one axis. */
    public int outDim(int in, int k, int s) {
        return this == SAME ? (in + s - 1) / s : (in - k) / s + 1;
    }

    /** Total padding to place BEFORE the window walk starts. */
    public int padBefore(int in, int k, int s) {
        if (this == VALID) return 0;
        int out = outDim(in, k, s);
        int total = Math.max(0, (out - 1) * s + k - in);
        return total / 2;                       // TF pads extra on the right/bottom
    }
}
