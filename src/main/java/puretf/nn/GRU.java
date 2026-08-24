package puretf.nn;

import puretf.Tensor;

/**
 * GRU (Cho et al. 2014), Keras-style z|r|h~ blocks:
 * <pre>
 *   z_t = sigma(x.k_z + h.u_z + b_z)
 *   r_t = sigma(x.k_r + h.u_r + b_r)
 *   h~  = tanh(x.k_h + r*(h.u_h) + b_h)
 *   h'  = (1-z)*h~ + z*h
 * </pre>
 */
public final class GRU extends RecurrentBase {

    public GRU(int units) { this(units, false); }

    public GRU(int units, boolean returnSequence) {
        super(units, returnSequence);
    }

    @Override
    protected int gateWidth() { return 3; }

    @Override
    protected Step step(Tensor xt, Tensor h, Tensor c) {
        int u = units;
        Tensor kz = kernel.sliceLast(0, u), kr = kernel.sliceLast(u, u),
                kh = kernel.sliceLast(2 * u, u);
        Tensor uz = recurrent.sliceLast(0, u),
                ur = recurrent.sliceLast(u, u),
                uh = recurrent.sliceLast(2 * u, u);
        Tensor bz = bias.sliceLast(0, u), br = bias.sliceLast(u, u),
                bh = bias.sliceLast(2 * u, u);

        Tensor hu = h.matmul(recurrent);          // shared projection
        Tensor zt = xt.matmul(kz).broadcastAdd(hu.sliceLast(0, u))
                .broadcastAdd(bz).sigmoid();
        Tensor rt = xt.matmul(kr).broadcastAdd(hu.sliceLast(u, u))
                .broadcastAdd(br).sigmoid();
        Tensor cand = xt.matmul(kh)
                .broadcastAdd(rt.mul(hu.sliceLast(2 * u, u)))
                .broadcastAdd(bh).tanh();
        Tensor hn = zt.scale(-1f).addScalar(1f).mul(cand)
                .broadcastAdd(zt.mul(h));
        return new Step(hn, null);
    }

    @Override public int typeCode() { return 25; }

    @Override public int[] specArgs() {
        return new int[]{units, returnSequence ? 1 : 0};
    }
}
