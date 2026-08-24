package puretf.nn;

import puretf.Tensor;

/** SELU (self-normalizing, Klambauer et al. 2017). */
public final class SELU extends ActivationLayer {

    @Override
    protected Tensor apply(Tensor x) { return x.selu(); }

    @Override public int typeCode() { return 7; }
}
