package puretf.nn;

import puretf.Tensor;

/** Hyperbolic tangent. */
public final class Tanh extends ActivationLayer {

    @Override
    protected Tensor apply(Tensor x) { return x.tanh(); }

    @Override public int typeCode() { return 12; }
}
