package puretf.nn;

import puretf.Tensor;

/** Vanilla tanh recurrent layer: h' = tanh(x.K + h.U + b). */
public final class RNN extends RecurrentBase {

    public RNN(int units) { this(units, false); }

    public RNN(int units, boolean returnSequence) {
        super(units, returnSequence);
    }

    @Override
    protected int gateWidth() { return 1; }

    @Override
    protected Step step(Tensor xt, Tensor h, Tensor c) {
        Tensor hn = xt.matmul(kernel)
                .broadcastAdd(h.matmul(recurrent))
                .broadcastAdd(bias)
                .tanh();
        return new Step(hn, null);
    }

    @Override public int typeCode() { return 23; }

    @Override public int[] specArgs() {
        return new int[]{units, returnSequence ? 1 : 0};
    }
}
