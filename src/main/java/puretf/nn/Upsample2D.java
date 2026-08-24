package puretf.nn;

import puretf.Tensor;

/** Nearest-neighbor upsampling by integer factors (NHWC). */
public final class Upsample2D extends ActivationLayer {

    private final int fh, fw;

    public Upsample2D(int factorH, int factorW) {
        this.fh = factorH;
        this.fw = factorW;
    }

    @Override
    protected Tensor apply(Tensor x) { return x.upsample2d(fh, fw); }

    @Override public int typeCode() { return 30; }

    @Override public int[] specArgs() { return new int[]{fh, fw}; }
}
