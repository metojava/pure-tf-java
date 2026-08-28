package puretf.nn;

import puretf.Tensor;

import java.util.List;
import java.util.Random;

/**
 * Multi-head scaled dot-product self-attention over [B,T,D] with residual
 * and optional causal mask:
 * <pre>
 *   per head h (dh = D/heads):
 *     scores_h = Q_h K_h&#7481;/&radic;dh          [B,T,T]
 *     ctx_h    = softmax(scores_h) V_h  [B,T,dh]
 *   concat(heads) . WO + x              [B,T,D]
 * </pre>
 * Heads are column slices of the shared [D,D] Q/K/V projections; each head's
 * slice of WO projects its context straight back to [B,T,D].
 */
public class MultiHeadAttention extends Layer {

    protected final int heads;
    protected final boolean causal;
    protected Tensor wq, wk, wv, wo;
    private final long seed;

    public MultiHeadAttention(int heads) { this(heads, false); }

    public MultiHeadAttention(int heads, boolean causal) {
        if (heads <= 0) throw new IllegalArgumentException("heads");
        this.heads = heads;
        this.causal = causal;
        this.seed = seedCounter++;
    }

    private static long seedCounter = 0;

    @Override
    protected void init(int fanIn) {
        if (fanIn % heads != 0)
            throw new IllegalArgumentException("D=" + fanIn
                    + " must divide by heads=" + heads);
        Random rnd = new Random(seed);
        int d = fanIn;
        wq = Tensor.glorot(rnd, d, d, d, d);
        wk = Tensor.glorot(rnd, d, d, d, d);
        wv = Tensor.glorot(rnd, d, d, d, d);
        wo = Tensor.glorot(rnd, d, d, d, d);
    }

    @Override
    protected Tensor apply(Tensor x) {
        int b = x.shape[0], t = x.shape[1], d = x.shape[2], dh = d / heads;
        Tensor q = proj(x, wq), k = proj(x, wk), v = proj(x, wv);
        float scale = (float) (1.0 / Math.sqrt(dh));

        Tensor sum = null;
        for (int h = 0; h < heads; h++) {
            Tensor qh = q.sliceLast(h * dh, dh);
            Tensor kh = k.sliceLast(h * dh, dh);
            Tensor vh = v.sliceLast(h * dh, dh);

            Tensor scores = qh.bmm(kh.permute(0, 2, 1)).scale(scale);
            if (causal) scores = scores.broadcastAdd(causalMask(t));
            Tensor ctx = scores.softmax().bmm(vh);          // [B,T,dh]

            Tensor p = ctx.reshape(b * t, dh)
                    .matmul(wo.rows(h * dh, dh))             // [B*T, D]
                    .reshape(b, t, d);
            sum = sum == null ? p : sum.broadcastAdd(p);
        }
        return sum.broadcastAdd(x);                          // residual
    }

    /** [B,T,D] . W[D,D] through the rank-2 matmul. */
    private static Tensor proj(Tensor x, Tensor w) {
        int b = x.shape[0], t = x.shape[1], d = x.shape[2];
        return x.reshape(b * t, d).matmul(w).reshape(b, t, d);
    }

    private static Tensor causalMask(int t) {
        float[] m = new float[t * t];
        for (int i = 0; i < t; i++)
            for (int j = i + 1; j < t; j++) m[i * t + j] = -1e9f;
        return Tensor.of(m, t, t);
    }

    @Override
    public List<Tensor> params() { return List.of(wq, wk, wv, wo); }

    public int heads() { return heads; }
    public boolean causal() { return causal; }

    @Override public int typeCode() { return 21; }

    @Override public int[] specArgs() { return new int[]{heads, causal ? 1 : 0}; }
}
