package puretf.nn;

import puretf.Tensor;

/** Max pooling (NHWC); padded cells are ignored. */
public final class MaxPool2D extends ActivationLayer {

    private final int kh, kw, sh, sw;
    private final Padding padding;

    public MaxPool2D(int kernelH, int kernelW, int strideH, int strideW,
                     Padding padding) {
        this.kh = kernelH;
        this.kw = kernelW;
        this.sh = strideH;
        this.sw = strideW;
        this.padding = padding;
    }

    @Override
    protected Tensor apply(Tensor x) {
        int h = x.shape[1], w = x.shape[2];
        return x.maxPool2d(kh, kw, sh, sw,
                padding.padBefore(h, kh, sh),
                padding.padBefore(w, kw, sw),
                padding.outDim(h, kh, sh),
                padding.outDim(w, kw, sw));
    }

    @Override public int typeCode() { return 28; }

    @Override public int[] specArgs() {
        return new int[]{kh, kw, sh, sw, padding.code};
    }
}
