package puretf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * A tiny define-by-run autograd tensor - the heart of the pure-Java engine.
 * Zero native code, zero dependencies.
 *
 * <p>Design (micrograd/tinynn style): every math operation creates a result
 * Tensor that remembers its parents and a backprop closure. {@link #backward()}
 walks the recorded tape in reverse and accumulates gradients into
 * {@code requiresGrad} leaves (parameters). Gradients live in {@code grad}.</p>
 *
 * <p>Training loop pattern:</p>
 * <pre>
 *   Tensor.resetGraph();                 // fresh tape for this step
 *   Node loss = ...build forward...;     // params are reused across steps
 *   param.zeroGrad();
 *   loss.backward();
 *   optimizer.step(params);
 * </pre>
 *
 * <p>Supported ops (each with analytic gradients): matmul, batched matmul,
 * bias-add / broadcast add, elementwise mul, reshape/permute, axis
 * mean/sum, scalar scale/add/power, relu/sigmoid/tanh/leaky-relu/elu/selu/
 * silu/gelu/softplus, softmax (last axis), embedding gather, sum/mean MSE,
 * fused softmax-cross-entropy. Broadcasting: NumPy-style right-aligned.</p>
 */
public final class Tensor {

    // ------------------------- tape (creation order) ---------------------
    private static final List<Tensor> TAPE = new ArrayList<>();

    /** Clears the autograd tape. Call once before each forward pass. */
    public static void resetGraph() {
        TAPE.clear();
    }

    // ------------------------------ fields -------------------------------
    public final float[] data;
    public final int[] shape;
    public float[] grad;                    // lazily allocated
    final boolean requiresGrad;             // params & activations: true
    private final List<Tensor> parents;     // null => leaf
    private Runnable backprop;              // pushes this.grad to parents

    private Tensor(float[] data, int[] shape, boolean requiresGrad,
                   List<Tensor> parents) {
        this.data = data;
        this.shape = shape;
        this.requiresGrad = requiresGrad;
        this.parents = parents;
        if (parents != null) TAPE.add(this);
    }

    private static Tensor op(float[] data, int[] shape, Tensor... parents) {
        Tensor t = new Tensor(data, shape, true,
                Arrays.asList(parents));
        return t;
    }

    private static Tensor leaf(float[] data, int[] shape, boolean requiresGrad) {
        return new Tensor(data, shape, requiresGrad, null);
    }

    private static void acc(Tensor t, float[] g) {
        if (!t.requiresGrad || g == null) return;
        if (t.grad == null) t.grad = new float[t.data.length];
        for (int i = 0; i < t.data.length; i++) t.grad[i] += g[i];
    }

    /** Escape hatch for custom layers building ops outside this class. */
    public static Tensor opPublic(float[] data, int[] shape, Tensor... parents) {
        return op(data, shape, parents);
    }

    /** Accumulates a gradient into a tensor (no-op for constants). */
    public static void accPublic(Tensor t, float[] g) {
        acc(t, g);
    }

    /** Escape hatch: attach a custom backprop closure to an op node. */
    public void setBackprop(Runnable r) {
        this.backprop = r;
    }

    // ---------------------------- factories ------------------------------

    /** Trainable parameter. */
    public static Tensor param(float[] data, int... shape) {
        return leaf(data, shape.clone(), true);
    }

    /** Constant / input placeholder (no gradient tracked). */
    public static Tensor of(float[] data, int... shape) {
        return leaf(data, shape.clone(), false);
    }

    public void zeroGrad() {
        if (grad != null) Arrays.fill(grad, 0f);
    }

    /** Seeds a scalar loss node with dL/dL = 1 and runs full backprop. */
    public void backward() {
        if (grad == null) grad = new float[data.length];
        Arrays.fill(grad, 0f);
        grad[0] += 1f;
        for (int i = TAPE.size() - 1; i >= 0; i--) {
            Tensor n = TAPE.get(i);
            if (n.backprop != null && n.grad != null) n.backprop.run();
        }
    }

    // ----------------------------- init ----------------------------------

    public static Tensor glorot(Random rnd, int fanIn, int fanOut, int... shape) {
        float limit = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        float[] data = new float[(int) product(shape)];
        for (int i = 0; i < data.length; i++)
            data[i] = (rnd.nextFloat() * 2f - 1f) * limit;
        return param(data, shape);
    }

    public static Tensor randn(Random rnd, float std, int... shape) {
        float[] data = new float[(int) product(shape)];
        for (int i = 0; i < data.length; i++) data[i] = (float) (rnd.nextGaussian() * std);
        return of(data, shape);
    }

    private static long product(int... shape) {
        long p = 1;
        for (int s : shape) p *= s;
        return p;
    }

    // ============================== OPS ==================================

    /** out[B,N] = A[B,M] . W[M,N] - batch rows split across threads. */
    public Tensor matmul(Tensor w) {
        int b = shape[0], m = shape[1], n = w.shape[1];
        float[] out = new float[b * n];
        if (Par.worth((long) b * m * n)) {
            Par.run((id, T) -> {
                int i0 = Par.start(id, b, T), i1 = Par.start(id + 1, b, T);
                for (int i = i0; i < i1; i++)
                    for (int k = 0; k < m; k++) {
                        float a = data[i * m + k];
                        if (a == 0f) continue;
                        int ao = i * m + k, oo = i * n;
                        for (int j = 0; j < n; j++)
                            out[oo + j] += a * w.data[k * n + j];
                    }
            });
        } else {
            for (int i = 0; i < b; i++)
                for (int k = 0; k < m; k++) {
                    float a = data[i * m + k];
                    if (a == 0f) continue;
                    int ao = i * m + k, oo = i * n;
                    for (int j = 0; j < n; j++)
                        out[oo + j] += a * w.data[k * n + j];
                }
        }
        Tensor r = op(out, new int[]{b, n}, this, w);
        r.backprop = () -> {
            // dA = G . W^T (rows of G partitioned); dW = A^T . G (rows of W
            // partitioned, each accumulating over the full batch in order)
            float[] da = new float[data.length];
            float[] dw = w.requiresGrad ? new float[m * n] : null;
            if (Par.worth((long) b * m * n)) {
                Par.run((id, T) -> {
                    int i0 = Par.start(id, b, T), i1 = Par.start(id + 1, b, T);
                    for (int i = i0; i < i1; i++)
                        for (int k = 0; k < m; k++) {
                            float s = 0;
                            int go = i * n, wo = k * n;
                            for (int j = 0; j < n; j++)
                                s += r.grad[go + j] * w.data[wo + j];
                            da[i * m + k] = s;
                        }
                    if (dw != null) {
                        int k0 = Par.start(id, m, T), k1 = Par.start(id + 1, m, T);
                        for (int k = k0; k < k1; k++)
                            for (int i = 0; i < b; i++) {
                                float a = data[i * m + k];
                                if (a == 0f) continue;
                                int ao = i * m + k, go = i * n, dwo = k * n;
                                for (int j = 0; j < n; j++)
                                    dw[dwo + j] += a * r.grad[go + j];
                            }
                    }
                });
            } else {
                for (int i = 0; i < b; i++)
                    for (int k = 0; k < m; k++) {
                        float s = 0;
                        int go = i * n, wo = k * n;
                        for (int j = 0; j < n; j++)
                            s += r.grad[go + j] * w.data[wo + j];
                        da[i * m + k] = s;
                    }
                if (dw != null)
                    for (int i = 0; i < b; i++)
                        for (int k = 0; k < m; k++) {
                            float a = data[i * m + k];
                            if (a == 0f) continue;
                            int ao = i * m + k, go = i * n, dwo = k * n;
                            for (int j = 0; j < n; j++)
                                dw[dwo + j] += a * r.grad[go + j];
                        }
            }
            acc(this, da);
            acc(w, dw);
        };
        return r;
    }

    /** out[B,N] = x[B,N] + b[N] (row-broadcast). */
    public Tensor addBias(Tensor bT) {
        int b = shape[0], n = shape[1];
        float[] out = new float[data.length];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < n; j++)
                out[i * n + j] = data[i * n + j] + bT.data[j];
        Tensor r = op(out, shape.clone(), this, bT);
        r.backprop = () -> {
            acc(this, r.grad);
            if (bT.requiresGrad) {
                float[] db = new float[n];
                for (int i = 0; i < b; i++)
                    for (int j = 0; j < n; j++)
                        db[j] += r.grad[i * n + j];
                acc(bT, db);
            }
        };
        return r;
    }

    /** Elementwise multiply, same shape. */
    public Tensor mul(Tensor o) {
        float[] out = new float[data.length];
        for (int i = 0; i < data.length; i++) out[i] = data[i] * o.data[i];
        Tensor r = op(out, shape.clone(), this, o);
        r.backprop = () -> {
            acc(this, mulArrays(r.grad, o.data));
            acc(o, mulArrays(r.grad, data));
        };
        return r;
    }

    public Tensor relu() {
        float[] out = new float[data.length];
        for (int i = 0; i < data.length; i++) out[i] = Math.max(0f, data[i]);
        Tensor r = op(out, shape.clone(), this);
        r.backprop = () -> {
            float[] g = new float[data.length];
            for (int i = 0; i < data.length; i++)
                g[i] = data[i] > 0 ? r.grad[i] : 0f;
            acc(this, g);
        };
        return r;
    }

    public Tensor sigmoid() {
        float[] out = new float[data.length];
        for (int i = 0; i < data.length; i++)
            out[i] = 1f / (1f + (float) Math.exp(-data[i]));
        Tensor r = op(out, shape.clone(), this);
        r.backprop = () -> {
            float[] g = new float[data.length];
            for (int i = 0; i < data.length; i++)
                g[i] = r.grad[i] * out[i] * (1f - out[i]);
            acc(this, g);
        };
        return r;
    }

    public Tensor tanh() {
        float[] out = new float[data.length];
        for (int i = 0; i < data.length; i++) out[i] = (float) Math.tanh(data[i]);
        Tensor r = op(out, shape.clone(), this);
        r.backprop = () -> {
            float[] g = new float[data.length];
            for (int i = 0; i < data.length; i++)
                g[i] = r.grad[i] * (1f - out[i] * out[i]);
            acc(this, g);
        };
        return r;
    }

    /** Mean squared error against target, averaged over ALL elements.
     *  Both shapes must match; returns scalar node. */
    public Tensor mse(Tensor target) {
        int m = data.length;
        double s = 0;
        for (int i = 0; i < m; i++) {
            double e = data[i] - target.data[i];
            s += e * e;
        }
        float loss = (float) (s / m);
        Tensor r = op(new float[]{loss}, new int[]{1}, this, target);
        r.backprop = () -> {
            float[] dp = new float[m];
            for (int i = 0; i < m; i++)
                dp[i] = r.grad[0] * 2f * (data[i] - target.data[i]) / m;
            acc(this, dp);
        };
        return r;
    }

    /**
     * Fused softmax cross-entropy over rows. logits [B,C], labels class ids.
     * Returns scalar mean loss; gradient w.r.t. logits is the classic
     * numerically-stable (softmax - onehot)/B.
     */
    public static Tensor softmaxCE(Tensor logits, int[] labels) {
        int b = logits.shape[0], c = logits.shape[1];
        float[] probs = new float[b * c];
        double loss = 0;
        for (int i = 0; i < b; i++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < c; j++)
                max = Math.max(max, logits.data[i * c + j]);
            double sum = 0;
            for (int j = 0; j < c; j++) {
                double e = Math.exp(logits.data[i * c + j] - max);
                probs[i * c + j] = (float) e;
                sum += e;
            }
            for (int j = 0; j < c; j++) probs[i * c + j] /= (float) sum;
            loss += -Math.log(Math.max(probs[i * c + labels[i]], 1e-12));
        }
        float mean = (float) (loss / b);
        Tensor r = op(new float[]{mean}, new int[]{1}, logits);
        r.backprop = () -> {
            float scale = r.grad[0] / b;
            float[] dl = new float[b * c];
            for (int i = 0; i < b; i++)
                for (int j = 0; j < c; j++)
                    dl[i * c + j] = scale * (probs[i * c + j]
                            - (labels[i] == j ? 1f : 0f));
            acc(logits, dl);
        };
        return r;
    }

    // ==================== SHAPE & BROADCASTING OPS =======================

    /** Reshape to arbitrary dimensions (same flat row-major order). */
    public Tensor reshape(int... newShape) {
        if (product(newShape) != data.length)
            throw new IllegalArgumentException("reshape "
                    + Arrays.toString(shape) + " -> "
                    + Arrays.toString(newShape) + ": size mismatch");
        Tensor r = op(data.clone(), newShape.clone(), this);
        r.backprop = () -> acc(this, r.grad);   // flat order preserved
        return r;
    }

    /** Dimension permutation, e.g. {@code [B,T,D] -> [B,D,T]} via (0,2,1). */
    public Tensor permute(int... perm) {
        int rank = shape.length;
        int[] outShape = new int[rank];
        for (int i = 0; i < rank; i++) outShape[i] = shape[perm[i]];
        int[] inStrides = strides(shape), outStrides = strides(outShape);
        float[] out = new float[(int) product(outShape)];
        int[] idx = new int[rank];
        for (int o = 0; o < out.length; o++) {
            int src = 0;
            for (int k = 0; k < rank; k++) src += idx[k] * inStrides[perm[k]];
            out[o] = data[src];
            for (int k = rank - 1; k >= 0; k--) {
                if (++idx[k] < outShape[k]) break;
                idx[k] = 0;
            }
        }
        Tensor r = op(out, outShape, this);
        r.backprop = () -> {
            float[] g = new float[data.length];
            int[] oi = new int[rank];
            for (int o = 0; o < r.grad.length; o++) {
                int dst = 0;
                for (int k = 0; k < rank; k++) dst += oi[k] * inStrides[perm[k]];
                g[dst] += r.grad[o];
                for (int k = rank - 1; k >= 0; k--) {
                    if (++oi[k] < outShape[k]) break;
                    oi[k] = 0;
                }
            }
            acc(this, g);
        };
        return r;
    }

    /** NumPy-style right-aligned broadcasting elementwise multiply
     *  (companion of {@link #broadcastAdd}). */
    public Tensor broadcastMul(Tensor o) {
        int rank = Math.max(shape.length, o.shape.length);
        int[] ad = padDims(shape, rank), bd = padDims(o.shape, rank);
        int[] od = new int[rank];
        for (int i = 0; i < rank; i++) {
            if (ad[i] != bd[i] && ad[i] != 1 && bd[i] != 1)
                throw new IllegalArgumentException("broadcastMul "
                        + Arrays.toString(shape) + " vs "
                        + Arrays.toString(o.shape));
            od[i] = Math.max(ad[i], bd[i]);
        }
        int[] as = zeroStrides(ad), bs = zeroStrides(bd);
        float[] out = new float[(int) product(od)];
        int[] fidx = new int[rank];
        for (int p = 0; p < out.length; p++) {
            int ia = 0, ib = 0;
            for (int k = 0; k < rank; k++) { ia += fidx[k] * as[k]; ib += fidx[k] * bs[k]; }
            out[p] = data[ia] * o.data[ib];
            for (int k = rank - 1; k >= 0; k--) {
                if (++fidx[k] < od[k]) break;
                fidx[k] = 0;
            }
        }
        Tensor r = op(out, od, this, o);
        r.backprop = () -> {
            int rk = od.length;
            int[] idx = new int[rk];
            float[] ga = this.requiresGrad ? new float[data.length] : null;
            float[] gb = o.requiresGrad ? new float[o.data.length] : null;
            for (int p = 0; p < r.grad.length; p++) {
                int ia = 0, ib = 0;
                for (int k = 0; k < rk; k++) { ia += idx[k] * as[k]; ib += idx[k] * bs[k]; }
                float g = r.grad[p];
                if (ga != null) ga[ia] += g * o.data[ib];
                if (gb != null) gb[ib] += g * data[ia];
                for (int k = rk - 1; k >= 0; k--) {
                    if (++idx[k] < od[k]) break;
                    idx[k] = 0;
                }
            }
            acc(this, ga);
            acc(o, gb);
        };
        return r;
    }

    /** NumPy-style right-aligned broadcasting elementwise add,
     *  e.g. {@code [B,N]+[N]}, {@code [B,T,D]+[D]}, {@code [B,H,W,C]+[C]}. */
    public Tensor broadcastAdd(Tensor o) {
        int rank = Math.max(shape.length, o.shape.length);
        int[] ad = padDims(shape, rank), bd = padDims(o.shape, rank);
        int[] od = new int[rank];
        for (int i = 0; i < rank; i++) {
            if (ad[i] != bd[i] && ad[i] != 1 && bd[i] != 1)
                throw new IllegalArgumentException("broadcastAdd "
                        + Arrays.toString(shape) + " vs "
                        + Arrays.toString(o.shape));
            od[i] = Math.max(ad[i], bd[i]);
        }
        int[] as = zeroStrides(ad), bs = zeroStrides(bd);
        float[] out = new float[(int) product(od)];
        int[] idx = new int[rank];
        for (int p = 0; p < out.length; p++) {
            int ia = 0, ib = 0;
            for (int k = 0; k < rank; k++) { ia += idx[k] * as[k]; ib += idx[k] * bs[k]; }
            out[p] = data[ia] + o.data[ib];
            for (int k = rank - 1; k >= 0; k--) {
                if (++idx[k] < od[k]) break;
                idx[k] = 0;
            }
        }
        Tensor r = op(out, od, this, o);
        r.backprop = () -> {
            acc(this, reduceBroadcast(r.grad, od, shape));
            acc(o, reduceBroadcast(r.grad, od, o.shape));
        };
        return r;
    }

    /** Mean over one axis (axis dropped from the result shape). */
    public Tensor meanAxis(int axis) {
        return reduceAxis(axis, true);
    }

    /** Sum over one axis (axis dropped from the result shape). */
    public Tensor sumAxis(int axis) {
        return reduceAxis(axis, false);
    }

    private Tensor reduceAxis(int axis, boolean mean) {
        int n = shape[axis], rank = shape.length;
        int outer = 1, inner = 1;
        for (int i = 0; i < axis; i++) outer *= shape[i];
        for (int i = axis + 1; i < rank; i++) inner *= shape[i];
        int[] outShape = new int[rank - 1];
        for (int i = 0, j = 0; i < rank; i++) if (i != axis) outShape[j++] = shape[i];
        float[] out = new float[outer * inner];
        for (int o = 0; o < outer; o++)
            for (int j = 0; j < n; j++) {
                int srcBase = (o * n + j) * inner;
                for (int i = 0; i < inner; i++) out[o * inner + i] += data[srcBase + i];
            }
        float f = mean ? 1f / n : 1f;
        for (int i = 0; i < out.length; i++) out[i] *= f;
        final int O = outer, I = inner;
        Tensor r = op(out, outShape, this);
        r.backprop = () -> {
            float[] g = new float[data.length];
            for (int o = 0; o < O; o++)
                for (int j = 0; j < n; j++) {
                    int dstBase = (o * n + j) * I;
                    for (int i = 0; i < I; i++) g[dstBase + i] += r.grad[o * I + i] * f;
                }
            acc(this, g);
        };
        return r;
    }

    // ===================== ELEMENTWISE SCALAR OPS ========================

    /** Multiply every element by a constant. */
    public Tensor scale(float s) {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) out[i] = data[i] * s;
        Tensor r = op(out, shape.clone(), this);
        r.backprop = () -> acc(this, scaleArray(r.grad, s));
        return r;
    }

    /** Add a constant to every element. */
    public Tensor addScalar(float c) {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) out[i] = data[i] + c;
        Tensor r = op(out, shape.clone(), this);
        r.backprop = () -> acc(this, r.grad);
        return r;
    }

    /** Elementwise {@code x^p} (covers sqrt with 0.5, rsqrt with -0.5). */
    public Tensor power(float p) {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) out[i] = (float) Math.pow(data[i], p);
        Tensor r = op(out, shape.clone(), this);
        r.backprop = () -> {
            float[] g = new float[data.length];
            for (int i = 0; i < g.length; i++)
                g[i] = r.grad[i] * p * (float) Math.pow(data[i], p - 1);
            acc(this, g);
        };
        return r;
    }

    // ========================== ACTIVATIONS ==============================

    /** Leaky ReLU: {@code x>0 ? x : alpha*x}. */
    public Tensor leakyRelu(float alpha) {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++)
            out[i] = data[i] > 0 ? data[i] : alpha * data[i];
        Tensor r = op(out, shape.clone(), this);
        r.backprop = () -> {
            float[] g = new float[data.length];
            for (int i = 0; i < g.length; i++)
                g[i] = r.grad[i] * (data[i] > 0 ? 1f : alpha);
            acc(this, g);
        };
        return r;
    }

    /** ELU: {@code x>0 ? x : alpha*(exp(x)-1)}. */
    public Tensor elu(float alpha) {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++)
            out[i] = data[i] > 0 ? data[i]
                    : alpha * (float) (Math.exp(data[i]) - 1);
        Tensor r = op(out, shape.clone(), this);
        r.backprop = () -> {
            float[] g = new float[data.length];
            for (int i = 0; i < g.length; i++)
                g[i] = r.grad[i] * (data[i] > 0 ? 1f : out[i] + alpha);
            acc(this, g);
        };
        return r;
    }

    /** SELU (self-normalizing, Klambauer et al. 2017). */
    public Tensor selu() {
        final float ALPHA = 1.6732632423543772f, LAMBDA = 1.0507009873554805f;
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++)
            out[i] = data[i] > 0 ? LAMBDA * data[i]
                    : LAMBDA * ALPHA * (float) (Math.exp(data[i]) - 1);
        Tensor r = op(out, shape.clone(), this);
        r.backprop = () -> {
            float[] g = new float[data.length];
            for (int i = 0; i < g.length; i++)
                g[i] = r.grad[i] * (data[i] > 0 ? LAMBDA : out[i] + LAMBDA * ALPHA);
            acc(this, g);
        };
        return r;
    }

    /** SiLU / swish: {@code x * sigmoid(x)}. */
    public Tensor silu() {
        float[] out = new float[data.length];
        float[] sig = new float[data.length];
        for (int i = 0; i < out.length; i++) {
            sig[i] = 1f / (1f + (float) Math.exp(-data[i]));
            out[i] = data[i] * sig[i];
        }
        Tensor r = op(out, shape.clone(), this);
        r.backprop = () -> {
            float[] g = new float[data.length];
            for (int i = 0; i < g.length; i++)
                g[i] = r.grad[i] * sig[i] * (1f + data[i] * (1f - sig[i]));
            acc(this, g);
        };
        return r;
    }

    /** GELU (tanh approximation, Hendrycks & Gimpel 2016). */
    public Tensor gelu() {
        final float C = 0.7978845608028654f;   // sqrt(2/pi)
        float[] out = new float[data.length];
        float[] t = new float[data.length];
        for (int i = 0; i < out.length; i++) {
            float x = data[i], x3 = x * x * x;
            t[i] = (float) Math.tanh(C * (x + 0.044715f * x3));
            out[i] = 0.5f * x * (1f + t[i]);
        }
        Tensor r = op(out, shape.clone(), this);
        r.backprop = () -> {
            float[] g = new float[data.length];
            for (int i = 0; i < g.length; i++) {
                float x = data[i];
                float dt = (1f - t[i] * t[i]) * C * (1f + 3f * 0.044715f * x * x);
                g[i] = r.grad[i] * (0.5f * (1f + t[i]) + 0.5f * x * dt);
            }
            acc(this, g);
        };
        return r;
    }

    /** Softplus: {@code log(1+exp(x))}, numerically stable. */
    public Tensor softplus() {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) {
            float x = data[i];
            out[i] = Math.max(x, 0f) + (float) Math.log(1 + Math.exp(-Math.abs(x)));
        }
        Tensor r = op(out, shape.clone(), this);
        r.backprop = () -> {
            float[] g = new float[data.length];
            for (int i = 0; i < g.length; i++)
                g[i] = r.grad[i] / (1f + (float) Math.exp(-data[i]));
            acc(this, g);
        };
        return r;
    }

    /** Softmax over the LAST axis (numerically stable). */
    public Tensor softmax() {
        int cols = shape[shape.length - 1], rows = data.length / cols;
        float[] out = new float[data.length];
        for (int i = 0; i < rows; i++) {
            int off = i * cols;
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < cols; j++) max = Math.max(max, data[off + j]);
            double sum = 0;
            for (int j = 0; j < cols; j++) {
                double e = Math.exp(data[off + j] - max);
                out[off + j] = (float) e;
                sum += e;
            }
            for (int j = 0; j < cols; j++) out[off + j] /= (float) sum;
        }
        Tensor r = op(out, shape.clone(), this);
        r.backprop = () -> {
            float[] g = new float[data.length];
            for (int i = 0; i < rows; i++) {
                int off = i * cols;
                float dot = 0;
                for (int j = 0; j < cols; j++) dot += r.grad[off + j] * out[off + j];
                for (int j = 0; j < cols; j++)
                    g[off + j] = out[off + j] * (r.grad[off + j] - dot);
            }
            acc(this, g);
        };
        return r;
    }

    // ====================== GATHER / BATCHED MATMUL ======================

    /** Contiguous slice along the LAST axis, any rank:
     *  {@code [..., D] -> [..., len]} starting at {@code offset}. */
    public Tensor sliceLast(int offset, int len) {
        int d = shape[shape.length - 1];
        if (offset < 0 || len <= 0 || offset + len > d)
            throw new IllegalArgumentException("sliceLast(" + offset + ","
                    + len + ") on last dim " + d);
        int rows = data.length / d;
        float[] out = new float[rows * len];
        for (int r = 0; r < rows; r++)
            System.arraycopy(data, r * d + offset, out, r * len, len);
        int[] outShape = shape.clone();
        outShape[outShape.length - 1] = len;
        Tensor r = op(out, outShape, this);
        r.backprop = () -> {
            if (!this.requiresGrad) return;
            if (this.grad == null) this.grad = new float[data.length];
            for (int rr = 0; rr < rows; rr++)
                for (int j = 0; j < len; j++)
                    this.grad[rr * d + offset + j] += r.grad[rr * len + j];
        };
        return r;
    }

    /** Contiguous block of ROWS of a rank-2 tensor: {@code [M,N] -> [len,N]}. */
    public Tensor rows(int offset, int len) {
        if (shape.length != 2 || offset < 0 || len <= 0 || offset + len > shape[0])
            throw new IllegalArgumentException("rows(" + offset + "," + len
                    + ") on " + Arrays.toString(shape));
        int n = shape[1];
        float[] out = new float[len * n];
        System.arraycopy(data, offset * n, out, 0, len * n);
        Tensor r = op(out, new int[]{len, n}, this);
        r.backprop = () -> {
            if (!this.requiresGrad) return;
            if (this.grad == null) this.grad = new float[data.length];
            for (int i = 0; i < len * n; i++)
                this.grad[offset * n + i] += r.grad[i];
        };
        return r;
    }

    /** Concatenates two tensors along the LAST axis (same leading dims). */
    public Tensor concatLast(Tensor o) {
        int da = shape[shape.length - 1], db = o.shape[o.shape.length - 1];
        int rowsA = data.length / da, rowsB = o.data.length / db;
        if (rowsA != rowsB || shape.length != o.shape.length)
            throw new IllegalArgumentException("concatLast "
                    + Arrays.toString(shape) + " vs " + Arrays.toString(o.shape));
        float[] out = new float[rowsA * (da + db)];
        for (int r = 0; r < rowsA; r++) {
            System.arraycopy(data, r * da, out, r * (da + db), da);
            System.arraycopy(o.data, r * db, out, r * (da + db) + da, db);
        }
        int[] outShape = shape.clone();
        outShape[outShape.length - 1] = da + db;
        Tensor r = op(out, outShape, this, o);
        r.backprop = () -> {
            if (this.requiresGrad) {
                if (this.grad == null) this.grad = new float[data.length];
                for (int rr = 0; rr < rowsA; rr++)
                    System.arraycopy(r.grad, rr * (da + db),
                            this.grad, rr * da, da);
            }
            if (o.requiresGrad) {
                if (o.grad == null) o.grad = new float[o.data.length];
                for (int rr = 0; rr < rowsB; rr++)
                    System.arraycopy(r.grad, rr * (da + db) + da,
                            o.grad, rr * db, db);
            }
        };
        return r;
    }

    /** Rows of {@code table[V,D]} gathered by class ids: out {@code [ids.length, D]}. */
    public static Tensor embedding(Tensor table, int[] ids) {
        int d = table.shape[table.shape.length - 1];
        float[] out = new float[ids.length * d];
        for (int i = 0; i < ids.length; i++)
            System.arraycopy(table.data, ids[i] * d, out, i * d, d);
        Tensor r = op(out, new int[]{ids.length, d}, table);
        r.backprop = () -> {
            if (!table.requiresGrad) return;
            if (table.grad == null) table.grad = new float[table.data.length];
            for (int i = 0; i < ids.length; i++)
                for (int j = 0; j < d; j++)
                    table.grad[ids[i] * d + j] += r.grad[i * d + j];
        };
        return r;
    }

    /** Batched matmul: {@code [B,M,K] . [B,K,N] -> [B,M,N]} - parallel over B. */
    public Tensor bmm(Tensor w) {
        int b = shape[0], m = shape[1], k = shape[2], n = w.shape[2];
        float[] out = new float[b * m * n];
        if (Par.worth((long) b * m * k * n)) {
            Par.run((id, T) -> {
                int b0 = Par.start(id, b, T), b1 = Par.start(id + 1, b, T);
                bmmRange(out, w.data, b0, b1, m, k, n);
            });
        } else {
            bmmRange(out, w.data, 0, b, m, k, n);
        }
        Tensor r = op(out, new int[]{b, m, n}, this, w);
        r.backprop = () -> {
            float[] da = new float[data.length];
            float[] dw = w.requiresGrad ? new float[w.data.length] : null;
            if (Par.worth((long) b * m * k * n)) {
                Par.run((id, T) -> {
                    int b0 = Par.start(id, b, T), b1 = Par.start(id + 1, b, T);
                    bmmBack(da, dw, r.grad, b0, b1, m, k, n);
                });
            } else {
                bmmBack(da, dw, r.grad, 0, b, m, k, n);
            }
            acc(this, da);
            acc(w, dw);
        };
        return r;
    }

    private void bmmRange(float[] out, float[] wd, int b0, int b1,
                          int m, int k, int n) {
        for (int i = b0; i < b1; i++)
            for (int p = 0; p < m; p++)
                for (int q = 0; q < k; q++) {
                    float a = data[i * m * k + p * k + q];
                    if (a == 0f) continue;
                    int wo = i * k * n + q * n, oo = i * m * n + p * n;
                    for (int j = 0; j < n; j++) out[oo + j] += a * wd[wo + j];
                }
    }

    private void bmmBack(float[] da, float[] dw, float[] g, float[] wd,
                         int b0, int b1, int m, int k, int n) {
        for (int i = b0; i < b1; i++) {
            int ao = i * m * k, wo = i * k * n, go = i * m * n;
            for (int p = 0; p < m; p++)
                for (int q = 0; q < k; q++) {
                    float ga = 0, a = data[ao + p * k + q];
                    int wOff = wo + q * n, gOff = go + p * n;
                    for (int j = 0; j < n; j++) {
                        float gv = g[gOff + j];
                        ga += gv * wd[wOff + j];
                        if (dw != null) dw[wOff + j] += a * gv;
                    }
                    da[ao + p * k + q] += ga;
                }
        }
    }

    // ============================ VISION OPS =============================

    /** Zero-pads a rank-4 NHWC tensor: pads {top, bottom, left, right}. */
    public Tensor pad2d(int pt, int pb, int pl, int pr) {
        if (shape.length != 4)
            throw new IllegalArgumentException("pad2d needs NHWC input");
        int b = shape[0], h = shape[1], w = shape[2], c = shape[3];
        int oh = h + pt + pb, ow = w + pl + pr;
        float[] out = new float[b * oh * ow * c];
        for (int i = 0; i < b; i++)
            for (int y = 0; y < h; y++) {
                int src = (i * h + y) * w * c;
                int dst = ((i * oh + y + pt) * ow + pl) * c;
                System.arraycopy(data, src, out, dst, w * c);
            }
        Tensor r = op(out, new int[]{b, oh, ow, c}, this);
        r.backprop = () -> {
            if (!this.requiresGrad) return;
            if (this.grad == null) this.grad = new float[data.length];
            for (int i = 0; i < b; i++)
                for (int y = 0; y < h; y++) {
                    int src = ((i * oh + y + pt) * ow + pl) * c;
                    int dst = (i * h + y) * w * c;
                    for (int k = 0; k < w * c; k++)
                        this.grad[dst + k] += r.grad[src + k];
                }
        };
        return r;
    }

    /**
     * im2col for NHWC convolution: window content laid out row-major
     * {@code [kh][kw][cin]} per output cell. Output
     * {@code [B*outH*outW, kh*kw*cin]}, ready for matmul with a
     * {@code [kh*kw*cin, filters]} kernel. Backward scatters (col2im).
     */
    public Tensor im2col(int kh, int kw, int sh, int sw, int pt, int pl,
                         int outH, int outW) {
        int b = shape[0], h = shape[1], w = shape[2], c = shape[3];
        int K = kh * kw * c;
        float[] out = new float[b * outH * outW * K];
        if (Par.worth((long) b * outH * outW * K)) {
            Par.run((id, T) -> {
                int b0 = Par.start(id, b, T), b1 = Par.start(id + 1, b, T);
                im2colRange(out, kh, kw, sh, sw, pt, pl, outH, outW, b0, b1);
            });
        } else {
            im2colRange(out, kh, kw, sh, sw, pt, pl, outH, outW, 0, b);
        }
        Tensor r = op(out, new int[]{b * outH * outW, K}, this);
        r.backprop = () -> {
            if (!this.requiresGrad) return;
            if (this.grad == null) this.grad = new float[data.length];
            if (Par.worth((long) b * outH * outW * K)) {
                Par.run((id, T) -> {
                    int b0 = Par.start(id, b, T), b1 = Par.start(id + 1, b, T);
                    col2imRange(r.grad, kh, kw, sh, sw, pt, pl, outH, outW,
                            b0, b1);
                });
            } else {
                col2imRange(r.grad, kh, kw, sh, sw, pt, pl, outH, outW, 0, b);
            }
        };
        return r;
    }

    private void im2colRange(float[] out, int kh, int kw, int sh, int sw,
                             int pt, int pl, int outH, int outW,
                             int b0, int b1) {
        int h = shape[1], w = shape[2], c = shape[3];
        int K = kh * kw * c;
        for (int i = b0; i < b1; i++)
            for (int oy = 0; oy < outH; oy++)
                for (int ox = 0; ox < outW; ox++) {
                    int dstBase = ((i * outH + oy) * outW + ox) * K;
                    for (int ky = 0; ky < kh; ky++)
                        for (int kx = 0; kx < kw; kx++) {
                            int iy = oy * sh - pt + ky;
                            int ix = ox * sw - pl + kx;
                            if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                            int src = ((i * h + iy) * w + ix) * c;
                            int colOff = (ky * kw + kx) * c;
                            System.arraycopy(data, src, out, dstBase + colOff, c);
                        }
                }
    }

    private void col2imRange(float[] g, int kh, int kw, int sh, int sw,
                             int pt, int pl, int outH, int outW,
                             int b0, int b1) {
        int h = shape[1], w = shape[2], c = shape[3];
        int K = kh * kw * c;
        for (int i = b0; i < b1; i++)
            for (int oy = 0; oy < outH; oy++)
                for (int ox = 0; ox < outW; ox++) {
                    int gBase = ((i * outH + oy) * outW + ox) * K;
                    for (int ky = 0; ky < kh; ky++)
                        for (int kx = 0; kx < kw; kx++) {
                            int iy = oy * sh - pt + ky;
                            int ix = ox * sw - pl + kx;
                            if (iy < 0 || iy >= h || ix < 0 || ix >= w)
                                continue;
                            int dst = ((i * h + iy) * w + ix) * c;
                            int colOff = (ky * kw + kx) * c;
                            for (int ch = 0; ch < c; ch++)
                                this.grad[dst + ch] += g[gBase + colOff + ch];
                        }
                }
    }

    /** Max pooling over NHWC, windows may straddle padding (excluded).
     *  Backward routes gradients to each window's max element. */
    public Tensor maxPool2d(int kh, int kw, int sh, int sw,
                            int pt, int pl, int outH, int outW) {
        int b = shape[0], h = shape[1], w = shape[2], c = shape[3];
        float[] out = new float[b * outH * outW * c];
        int[][] argmax = new int[out.length / c][];
        for (int i = 0, o = 0; i < b; i++)
            for (int oy = 0; oy < outH; oy++)
                for (int ox = 0; ox < outW; ox++, o++) {
                    int[] am = new int[c];
                    java.util.Arrays.fill(am, -1);
                    int outBase = o * c;
                    for (int ky = 0; ky < kh; ky++)
                        for (int kx = 0; kx < kw; kx++) {
                            int iy = oy * sh - pt + ky;
                            int ix = ox * sw - pl + kx;
                            if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                            int src = ((i * h + iy) * w + ix) * c;
                            for (int ch = 0; ch < c; ch++)
                                if (am[ch] == -1
                                        || data[src + ch] > data[am[ch]]) {
                                    am[ch] = src + ch;
                                    out[outBase + ch] = data[src + ch];
                                }
                        }
                    argmax[o] = am;
                }
        Tensor r = op(out, new int[]{b, outH, outW, c}, this);
        final int[][] AM = argmax;
        r.backprop = () -> {
            if (!this.requiresGrad) return;
            if (this.grad == null) this.grad = new float[data.length];
            for (int oIdx = 0; oIdx < AM.length; oIdx++) {
                int[] am = AM[oIdx];
                if (am == null) continue;
                for (int ch = 0; ch < c; ch++)
                    if (am[ch] >= 0)
                        this.grad[am[ch]] += r.grad[oIdx * c + ch];
            }
        };
        return r;
    }

    /** Average pooling over NHWC; padded cells excluded from the divisor. */
    public Tensor avgPool2d(int kh, int kw, int sh, int sw,
                            int pt, int pl, int outH, int outW) {
        int b = shape[0], h = shape[1], w = shape[2], c = shape[3];
        float[] out = new float[b * outH * outW * c];
        int[] counts = new int[out.length / c];
        for (int i = 0, o = 0; i < b; i++)
            for (int oy = 0; oy < outH; oy++)
                for (int ox = 0; ox < outW; ox++, o++) {
                    int cnt = 0;
                    int outBase = o * c;
                    for (int ky = 0; ky < kh; ky++)
                        for (int kx = 0; kx < kw; kx++) {
                            int iy = oy * sh - pt + ky;
                            int ix = ox * sw - pl + kx;
                            if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                            cnt++;
                            int src = ((i * h + iy) * w + ix) * c;
                            for (int ch = 0; ch < c; ch++)
                                out[outBase + ch] += data[src + ch];
                        }
                    counts[o] = Math.max(cnt, 1);
                    for (int ch = 0; ch < c; ch++) out[outBase + ch] /= cnt;
                }
        Tensor r = op(out, new int[]{b, outH, outW, c}, this);
        final int[] CNT = counts;
        r.backprop = () -> {
            if (!this.requiresGrad) return;
            if (this.grad == null) this.grad = new float[data.length];
            for (int i = 0, o = 0; i < b; i++)
                for (int oy = 0; oy < outH; oy++)
                    for (int ox = 0; ox < outW; ox++, o++) {
                        float inv = 1f / CNT[o];
                        for (int ky = 0; ky < kh; ky++)
                            for (int kx = 0; kx < kw; kx++) {
                                int iy = oy * sh - pt + ky;
                                int ix = ox * sw - pl + kx;
                                if (iy < 0 || iy >= h || ix < 0 || ix >= w)
                                    continue;
                                int dst = ((i * h + iy) * w + ix) * c;
                                for (int ch = 0; ch < c; ch++)
                                    this.grad[dst + ch] +=
                                            r.grad[o * c + ch] * inv;
                            }
                    }
        };
        return r;
    }

    /** Nearest-neighbor upsampling by integer factors (NHWC). */
    public Tensor upsample2d(int fh, int fw) {
        int b = shape[0], h = shape[1], w = shape[2], c = shape[3];
        int oh = h * fh, ow = w * fw;
        float[] out = new float[b * oh * ow * c];
        for (int i = 0; i < b; i++)
            for (int oy = 0; oy < oh; oy++)
                for (int ox = 0; ox < ow; ox++) {
                    int src = ((i * h + oy / fh) * w + ox / fw) * c;
                    int dst = ((i * oh + oy) * ow + ox) * c;
                    System.arraycopy(data, src, out, dst, c);
                }
        Tensor r = op(out, new int[]{b, oh, ow, c}, this);
        r.backprop = () -> {
            if (!this.requiresGrad) return;
            if (this.grad == null) this.grad = new float[data.length];
            for (int i = 0; i < b; i++)
                for (int oy = 0; oy < oh; oy++)
                    for (int ox = 0; ox < ow; ox++) {
                        int dst = ((i * oh + oy) * ow + ox) * c;
                        int src = ((i * h + oy / fh) * w + ox / fw) * c;
                        for (int ch = 0; ch < c; ch++)
                            this.grad[src + ch] += r.grad[dst + ch];
                    }
        };
        return r;
    }

    // --------------------------- helpers ---------------------------------

    private static float[] mulArrays(float[] a, float[] b) {
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) out[i] = a[i] * b[i];
        return out;
    }

    private static float[] scaleArray(float[] a, float s) {
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) out[i] = a[i] * s;
        return out;
    }

    private static int[] strides(int[] dims) {
        int[] st = new int[dims.length];
        st[dims.length - 1] = 1;
        for (int i = dims.length - 2; i >= 0; i--) st[i] = st[i + 1] * dims[i + 1];
        return st;
    }

    private static int[] zeroStrides(int[] dims) {
        int[] st = strides(dims);
        for (int i = 0; i < dims.length; i++) if (dims[i] == 1 && st.length > 1) st[i] = 0;
        return st;
    }

    private static int[] padDims(int[] dims, int rank) {
        int[] out = new int[rank];
        java.util.Arrays.fill(out, 1);
        System.arraycopy(dims, 0, out, rank - dims.length, dims.length);
        return out;
    }

    /** Sums an outer-shaped gradient down to the target shape (broadcast adjoint). */
    private static float[] reduceBroadcast(float[] g, int[] outDims, int[] targetDims) {
        int rank = outDims.length;
        int[] ts = zeroStrides(padDims(targetDims, rank));
        float[] out = new float[(int) product(targetDims)];
        int[] idx = new int[rank];
        for (int o = 0; o < g.length; o++) {
            int dst = 0;
            for (int k = 0; k < rank; k++) dst += idx[k] * ts[k];
            out[dst] += g[o];
            for (int k = rank - 1; k >= 0; k--) {
                if (++idx[k] < outDims[k]) break;
                idx[k] = 0;
            }
        }
        return out;
    }

    @Override
    public String toString() {
        return "Tensor" + Arrays.toString(shape)
                + (grad != null ? " (grad set)" : "");
    }
}
