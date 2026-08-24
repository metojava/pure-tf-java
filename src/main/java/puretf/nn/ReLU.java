package puretf.nn;

import puretf.Tensor;

/** Rectified linear unit. */
public final class ReLU extends ActivationLayer {

    @Override
    protected Tensor apply(Tensor x) { return x.relu(); }

    @Override public int typeCode() { return 4; }
}
