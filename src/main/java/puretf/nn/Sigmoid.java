package puretf.nn;

import puretf.Tensor;

/** Logistic sigmoid. */
public final class Sigmoid extends ActivationLayer {

    @Override
    protected Tensor apply(Tensor x) { return x.sigmoid(); }

    @Override public int typeCode() { return 11; }
}
