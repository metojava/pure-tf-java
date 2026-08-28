package puretf.nn;

import puretf.Tensor;

/**
 * Reshape to fixed dimensions; one dimension may be {@code -1} and is
 * inferred from the element count.
 */
public final class Reshape extends ActivationLayer {

    private final int[] target;

    public Reshape(int... target) { this.target = target.clone(); }

    @Override
    protected Tensor apply(Tensor x) {
        int[] out = target.clone();
        long known = 1;
        int neg = -1;
        for (int i = 0; i < out.length; i++) {
            if (out[i] == -1) {
                if (neg != -1) throw new IllegalArgumentException(
                        "Reshape: only one -1 allowed");
                neg = i;
            } else known *= out[i];
        }
        if (neg != -1) out[neg] = (int) (x.data.length / known);
        return x.reshape(out);
    }

    @Override public int typeCode() { return 14; }

    @Override public int[] specArgs() { return target.clone(); }
}
