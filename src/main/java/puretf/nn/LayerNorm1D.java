package puretf.nn;

import puretf.Tensor;

import java.util.List;

/**
 * Layer normalization over the LAST axis (Ba et al. 2016). Works on any
 * rank: [B,D] normalizes per sample, [B,T,D] per time step. Composed from
 * primitive autograd ops, so gradients are correct by construction.
 */
public final class LayerNorm1D extends Layer {

    private final float eps;
    private Tensor gamma, beta;

    public LayerNorm1D() { this(1e-5f); }

    public LayerNorm1D(float eps) { this.eps = eps; }

    @Override
    protected void init(int fanIn) {
        float[] ones = new float[fanIn];
        java.util.Arrays.fill(ones, 1f);
        gamma = Tensor.param(ones, fanIn);
        beta = Tensor.param(new float[fanIn], fanIn);
    }

    @Override
    protected Tensor apply(Tensor x) {
        int rank = x.shape.length;
        int[] keep = x.shape.clone();
        keep[rank - 1] = 1;
        Tensor mu = x.meanAxis(rank - 1).reshape(keep).scale(-1f);
        Tensor xc = x.broadcastAdd(mu);
        Tensor var = xc.mul(xc).meanAxis(rank - 1).reshape(keep);
        Tensor invStd = var.addScalar(eps).power(-0.5f);
        return xc.broadcastMul(invStd).broadcastMul(gamma).broadcastAdd(beta);
    }

    @Override
    public List<Tensor> params() { return List.of(gamma, beta); }

    @Override public int typeCode() { return 16; }

    @Override public int[] specArgs() { return new int[]{Math.round(eps * 1e7f)}; }
}
