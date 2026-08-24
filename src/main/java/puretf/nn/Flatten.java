package puretf.nn;

import puretf.Tensor;

/** Flattens all axes after the batch dim: [B, ...] -> [B, rest]. */
public final class Flatten extends ActivationLayer {

    @Override
    protected Tensor apply(Tensor x) {
        int b = x.shape[0];
        return x.reshape(b, x.data.length / b);
    }

    @Override public int typeCode() { return 3; }
}
