package puretf.nn;

import puretf.Tensor;

/** GELU activation (tanh approximation, Hendrycks & Gimpel 2016). */
public final class GELU extends ActivationLayer {

    @Override
    protected Tensor apply(Tensor x) { return x.gelu(); }

    @Override public int typeCode() { return 9; }
}
