package puretf.nn;

import puretf.Tensor;

/** Softplus: {@code log(1+exp(x))}. */
public final class Softplus extends ActivationLayer {

    @Override
    protected Tensor apply(Tensor x) { return x.softplus(); }

    @Override public int typeCode() { return 10; }
}
