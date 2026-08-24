package puretf.nn;

import puretf.Tensor;

import java.util.List;
import java.util.Random;

/**
 * Shared machinery for the recurrent family. Input [B,T,D]; each time step
 * is unrolled through primitive autograd ops (BPTT through the tape), so
 * gradients are exact. Output is the last hidden state [B,H] or, with
 * {@code returnSequence}, all states stacked [B,T,H].
 */
abstract class RecurrentBase extends Layer {

    protected final int units;
    protected final boolean returnSequence;
    protected Tensor kernel, recurrent, bias;
    private final long seed;

    /** Gate width per step: 1 for tanh-RNN/GRU-style 3 blocks, LSTM 4. */
    protected abstract int gateWidth();

    RecurrentBase(int units, boolean returnSequence) {
        this.units = units;
        this.returnSequence = returnSequence;
        this.seed = seedCounter++;
    }

    private static long seedCounter = 0;

    @Override
    protected void init(int fanIn) {
        Random rnd = new Random(seed);
        int k = gateWidth();
        kernel = Tensor.glorot(rnd, fanIn, units * k,
                fanIn, units * k);
        recurrent = Tensor.glorot(rnd, units, units * k,
                units, units * k);
        bias = Tensor.param(new float[units * k], units * k);
        initBiasHook();
    }

    /** Hook for special bias inits (LSTM forget gate = 1). */
    protected void initBiasHook() {}

    @Override
    protected Tensor apply(Tensor x) {
        int b = x.shape[0], t = x.shape[1], d = x.shape[2];
        // [B,T,D] -> [T,B,D] -> [T*B,D]: step i lives in rows [i*b,(i+1)*b)
        Tensor flat = x.permute(1, 0, 2).reshape(t * b, d);
        Tensor h = zeros(b);
        Tensor c = zeros(b);
        Tensor seq = null;

        for (int i = 0; i < t; i++) {
            Tensor xt = flat.rows(i * b, b);
            Step s = step(xt, h, c);
            h = s.h;
            c = s.c;
            if (returnSequence)
                seq = seq == null ? h : seq.concatLast(h);
        }
        if (returnSequence) {
            // [B,T] frames of H -> [B,T,H]
            return seq.reshape(b, t, units);
        }
        return h;
    }

    /** One recurrence step; implementations update gates from (xt,h,c). */
    protected abstract Step step(Tensor xt, Tensor h, Tensor c);

    protected static final class Step {
        final Tensor h, c;
        Step(Tensor h, Tensor c) { this.h = h; this.c = c; }
    }

    private Tensor zeros(int b) {
        return Tensor.of(new float[b * units], b, units);
    }

    @Override
    public List<Tensor> params() { return List.of(kernel, recurrent, bias); }

    public int units() { return units; }
}
