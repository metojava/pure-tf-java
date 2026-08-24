package puretf.nn;

import puretf.Tensor;

/** Mean over the time/sequence axis: [B,T,D] -> [B,D]. */
public final class GlobalAveragePooling extends ActivationLayer {

    @Override
    protected Tensor apply(Tensor x) { return x.meanAxis(1); }

    @Override public int typeCode() { return 15; }
}
