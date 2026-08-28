package puretf.nn;

import puretf.Tensor;

/**
 * LSTM (Hochreiter & Schmidhuber 1997) with Keras-style [D,4H] kernel and
 * i|f|g|o gate blocks; forget-gate bias initialized to 1.
 * <pre>
 *   z    = x.K + h.U + b                    [B,4H]
 *   c'   = sigma(z_f)*c + sigma(z_i)*tanh(z_g)
 *   h'   = sigma(z_o)*tanh(c')
 * </pre>
 */
public final class LSTM extends RecurrentBase {

    public LSTM(int units) { this(units, false); }

    public LSTM(int units, boolean returnSequence) {
        super(units, returnSequence);
    }

    @Override
    protected int gateWidth() { return 4; }

    @Override
    protected void initBiasHook() {
        int u = units;
        for (int i = 0; i < u; i++) bias.data[u + i] = 1f;   // forget block
    }

    @Override
    protected Step step(Tensor xt, Tensor h, Tensor c) {
        int u = units;
        Tensor z = xt.matmul(kernel)
                .broadcastAdd(h.matmul(recurrent))
                .broadcastAdd(bias);
        Tensor gi = z.sliceLast(0, u).sigmoid();
        Tensor gf = z.sliceLast(u, u).sigmoid();
        Tensor gg = z.sliceLast(2 * u, u).tanh();
        Tensor go = z.sliceLast(3 * u, u).sigmoid();
        Tensor cn = gf.mul(c).broadcastAdd(gi.mul(gg));
        Tensor hn = go.mul(cn.tanh());
        return new Step(hn, cn);
    }

    @Override public int typeCode() { return 24; }

    @Override public int[] specArgs() {
        return new int[]{units, returnSequence ? 1 : 0};
    }
}
