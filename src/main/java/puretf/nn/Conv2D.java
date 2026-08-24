package puretf.nn;

import puretf.Tensor;

import java.util.List;
import java.util.Random;

/**
 * 2-D convolution (NHWC) with bias via im2col + matmul:
 * filter {@code [kh, kw, cin, filters]}, stride {@code [sh, sw]},
 * Glorot-uniform init.
 */
public final class Conv2D extends Layer {

    private final int filters, kh, kw, sh, sw;
    private final Padding padding;
    private Tensor w, b;

    public Conv2D(int filters, int kernelH, int kernelW,
                  int strideH, int strideW, Padding padding) {
        this.filters = filters;
        this.kh = kernelH;
        this.kw = kernelW;
        this.sh = strideH;
        this.sw = strideW;
        this.padding = padding;
    }

    @Override
    protected void init(int fanIn) {
        Random rnd = new Random(seedCounter++);
        int K = kh * kw * fanIn;
        w = Tensor.glorot(rnd, K, filters, K, filters);
        b = Tensor.param(new float[filters], filters);
    }

    @Override
    protected Tensor apply(Tensor x) {
        int cin = x.shape[3];
        int h = x.shape[1], wd = x.shape[2];
        int oh = padding.outDim(h, kh, sh);
        int ow = padding.outDim(wd, kw, sw);
        Tensor cols = x.im2col(kh, kw, sh, sw,
                padding.padBefore(h, kh, sh),
                padding.padBefore(wd, kw, sw),
                oh, ow);
        Tensor y = cols.matmul(w).addBias(b);
        return y.reshape(x.shape[0], oh, ow, filters);
    }

    @Override
    public List<Tensor> params() { return List.of(w, b); }

    public int filters() { return filters; }

    @Override public int typeCode() { return 26; }

    @Override public int[] specArgs() {
        return new int[]{filters, kh, kw, sh, sw, padding.code};
    }

    private static long seedCounter = 0;
}
