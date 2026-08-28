package puretf.nn;

import puretf.Tensor;

/** SiLU / swish: {@code x * sigmoid(x)}. */
public final class SiLU extends ActivationLayer {

    @Override
    protected Tensor apply(Tensor x) { return x.silu(); }

    @Override public int typeCode() { return 8; }
}
