package puretf.nn;

import puretf.Tensor;

/** Leaky ReLU: {@code x>0 ? x : alpha*x}. */
public final class LeakyReLU extends ActivationLayer {

    private final float alpha;

    public LeakyReLU() { this(0.01f); }

    public LeakyReLU(float alpha) { this.alpha = alpha; }

    @Override
    protected Tensor apply(Tensor x) { return x.leakyRelu(alpha); }

    @Override public int typeCode() { return 5; }

    @Override public int[] specArgs() { return new int[]{Math.round(alpha * 10000)}; }
}
