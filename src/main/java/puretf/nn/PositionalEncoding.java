package puretf.nn;

import puretf.Tensor;

/**
 * Sinusoidal positional encoding (Vaswani et al. 2017): adds fixed
 * position-dependent signals to a [B,T,D] sequence so attention layers can
 * distinguish token order. Parameter-free.
 */
public final class PositionalEncoding extends ActivationLayer {

    @Override
    protected Tensor apply(Tensor x) {
        int t = x.shape[x.shape.length - 2], d = x.shape[x.shape.length - 1];
        float[] pe = new float[t * d];
        for (int pos = 0; pos < t; pos++)
            for (int i = 0; i < d; i++) {
                double angle = pos / Math.pow(10000, (i / (double) d) * 2);
                pe[pos * d + i] = (float) (i % 2 == 0 ? Math.sin(angle)
                        : Math.cos(angle));
            }
        return x.broadcastAdd(Tensor.of(pe, x.shape[1], d));
    }

    @Override public int typeCode() { return 19; }
}
