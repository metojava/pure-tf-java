package puretf.nn;

import puretf.Tensor;

/** Exponential linear unit: {@code x>0 ? x : alpha*(exp(x)-1)}. */
public final class ELU extends ActivationLayer {

    private final float alpha;

    public ELU() { this(1f); }

    public ELU(float alpha) { this.alpha = alpha; }

    @Override
    protected Tensor apply(Tensor x) { return x.elu(alpha); }

    @Override public int typeCode() { return 6; }

    @Override public int[] specArgs() { return new int[]{Math.round(alpha * 1000)}; }
}
