package puretf.nn;

import puretf.Optim;
import puretf.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Keras-style container: stack layers, then fit() / fitClassifier() /
 * fitHistory(). Weights initialize lazily on the first batch - no input dim
 * needed. Models persist via saveTo()/loadFrom().
 */
public final class Sequential {

    private final List<Layer> layers = new ArrayList<>();

    public Sequential add(Layer l) {
        layers.add(l);
        return this;
    }

    /** Switches Dropout-style layers between train and inference mode. */
    public void setTraining(boolean training) {
        for (Layer l : layers) l.setTraining(training);
    }

    public List<Tensor> params() {
        List<Tensor> ps = new ArrayList<>();
        for (Layer l : layers) ps.addAll(l.params());
        return ps;
    }

    private int firstInputDim = -1;

    /** The layers in order (unmodifiable view). */
    public List<Layer> layers() { return java.util.Collections.unmodifiableList(layers); }

    /**
     * Forces lazy weight initialization by running one dummy forward pass.
     * Layers that need a concrete input width become ready afterwards.
     */
    public void build(int inputDim) {
        forward(Tensor.of(new float[inputDim], 1, inputDim));
    }

    /**
     * Keras-style model summary: layer index, type, and the number of
     * trainable parameters per layer, plus a total. Call {@link #build(int)}
     * first so lazy weights exist; otherwise only reports what is initialized.
     */
    public void summary() {
        long total = 0;
        boolean anyUninit = false;
        System.out.printf("%-3s %-24s %14s%n", "Idx", "Layer (type)", "Params");
        System.out.println("-".repeat(45));
        for (int i = 0; i < layers.size(); i++) {
            Layer l = layers.get(i);
            if (!l.initialized()) { anyUninit = true; continue; }
            long c = 0;
            for (Tensor p : l.params()) c += p.data.length;
            total += c;
            System.out.printf("%-3d %-24s %14d%n", i, l.typeName(), c);
        }
        System.out.println("-".repeat(45));
        System.out.printf("%-27s %14d%n", "Total params", total);
        if (anyUninit)
            System.out.println("Note: some layers are not initialized yet. "
                    + "Call build(inputDim) or train once before summary().");
    }

    public Tensor forward(Tensor x) {
        if (firstInputDim == -1 && x.data.length > 0 && x.shape[0] > 0)
            firstInputDim = x.data.length / x.shape[0];
        for (Layer l : layers) x = l.forward(x);
        return x;
    }

    /** Regression training (MSE). Returns per-epoch mean losses. */
    public double[] fit(float[][] X, float[][] Y, int epochs, int batch, float lr) {
        return fit(X, Y, epochs, batch, lr, false);
    }

    /** Regression training with per-epoch progress logging. */
    public double[] fit(float[][] X, float[][] Y, int epochs, int batch,
                        float lr, boolean verbose) {
        Optim.Adam adam = new Optim.Adam(lr);
        forward(Tensor.of(X[0], 1, X[0].length));   // lazy-init layers
        List<Tensor> params = params();
        int n = X.length, d = X[0].length;
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Random shuf = new Random(7L);
        double[] hist = new double[epochs];

        for (int epoch = 0; epoch < epochs; epoch++) {
            shuffle(idx, shuf);
            long t0 = System.nanoTime();
            double sum = 0;
            int seen = 0;
            for (int start = 0; start < n; start += batch) {
                int len = Math.min(batch, n - start);
                Tensor.resetGraph();
                float[] bx = new float[len * d];
                float[] by = new float[len * Y[0].length];
                for (int i = 0; i < len; i++) {
                    System.arraycopy(X[idx[start + i]], 0, bx, i * d, d);
                    System.arraycopy(Y[idx[start + i]], 0, by, i * Y[0].length, Y[0].length);
                }
                var loss = forward(Tensor.of(bx, len, d)).mse(Tensor.of(by, len, Y[0].length));
                for (Tensor p : params) p.zeroGrad();
                loss.backward();
                adam.step(params);
                hist[epoch] += loss.data[0] * len;
                sum += loss.data[0] * len;
                seen += len;
            }
            hist[epoch] = sum / Math.max(seen, 1);
            if (verbose)
                System.out.printf("Epoch %d/%d - loss %.4f (%.1f s)%n",
                        epoch + 1, epochs, hist[epoch],
                        (System.nanoTime() - t0) / 1e9);
            System.out.flush();   // realtime progress, even when piped
        }
        return hist;
    }

    /** Classification training (fused softmax cross-entropy on logits). */
    public double[] fitClassifier(float[][] X, int[] Y, int classes,
                                  int epochs, int batch, float lr) {
        return fitClassifier(X, Y, classes, epochs, batch, lr, false).loss();
    }

    /** Classification training returning full Keras-style curves. */
    public History fitClassifier(float[][] X, int[] Y, int classes,
                                 int epochs, int batch, float lr,
                                 boolean verbose) {
        return fitHistory(X, Y, classes, epochs, batch,
                new Optim.Adam(lr), LrSchedule.constant(lr),
                0f, null, verbose);
    }

    /**
     * Full Keras-style trainer: optimizer + LR schedule, optional validation
     * split (last fraction of a seeded shuffle), early stopping with
     * best-weights restore. Returns the per-epoch curves.
     */
    public History fitHistory(float[][] X, int[] Y, int classes,
                              int epochs, int batch,
                              Optim.Stepper opt, LrSchedule sched,
                              float valSplit, EarlyStopping earlyStop,
                              boolean verbose) {
        forward(Tensor.of(X[0], 1, X[0].length));   // lazy-init layers
        List<Tensor> params = params();
        int d = X[0].length;

        // seeded split: tail of a shuffled index list becomes validation
        Integer[] order = new Integer[X.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        java.util.Collections.shuffle(java.util.Arrays.asList(order),
                new Random(7L));
        int nVal = Math.round(X.length * valSplit);
        int nTr = X.length - nVal;
        float[][] trX = new float[nTr][], valX = null;
        int[] trY = new int[nTr], valY = null;
        for (int i = 0; i < nTr; i++) {
            trX[i] = X[order[i]];
            trY[i] = Y[order[i]];
        }
        if (nVal > 0) {
            valX = new float[nVal][];
            valY = new int[nVal];
            for (int i = 0; i < nVal; i++) {
                valX[i] = X[order[nTr + i]];
                valY[i] = Y[order[nTr + i]];
            }
        }

        double[] lossH = new double[epochs], accH = new double[epochs];
        double[] vlH = nVal > 0 ? new double[epochs] : null;
        double[] vaH = nVal > 0 ? new double[epochs] : null;
        List<float[]> bestWeights = null;
        boolean stopped = false;
        int done = 0;

        for (int epoch = 0; epoch < epochs && !stopped; epoch++) {
            opt.setLr(sched.lrAt(epoch));
            setTraining(true);
            long t0 = System.nanoTime();
            double sum = 0;
            int seen = 0, correct = 0;
            int[] idx = new int[nTr];
            for (int i = 0; i < nTr; i++) idx[i] = i;
            shuffle(idx, new Random(1000L + epoch));
            for (int start = 0; start < nTr; start += batch) {
                int len = Math.min(batch, nTr - start);
                Tensor.resetGraph();
                float[] bx = new float[len * d];
                int[] by = new int[len];
                for (int i = 0; i < len; i++) {
                    System.arraycopy(trX[idx[start + i]], 0, bx, i * d, d);
                    by[i] = trY[idx[start + i]];
                }
                Tensor logits = forward(Tensor.of(bx, len, d));
                var loss = Tensor.softmaxCE(logits, by);
                for (Tensor p : params) p.zeroGrad();
                loss.backward();
                opt.step(params);
                for (int i = 0; i < len; i++)
                    if (argmaxAt(logits.data, i * classes, classes) == by[i]) correct++;
                sum += loss.data[0] * len;
                seen += len;
            }
            lossH[epoch] = sum / Math.max(seen, 1);
            accH[epoch] = correct / (double) seen;

            double metric;
            if (nVal > 0) {
                double[] ve = evaluate(valX, valY, classes);
                vlH[epoch] = ve[0];
                vaH[epoch] = ve[1];
                metric = earlyStop != null
                        && earlyStop.monitor == EarlyStopping.Monitor.ACCURACY
                        ? ve[1] : ve[0];
            } else {
                metric = lossH[epoch];
            }

            if (verbose) {
                StringBuilder sb = new StringBuilder(String.format(
                        "Epoch %d/%d - loss %.4f - accuracy %.4f",
                        epoch + 1, epochs, lossH[epoch], accH[epoch]));
                if (nVal > 0)
                    sb.append(String.format(" - val_loss %.4f - val_acc %.4f",
                            vlH[epoch], vaH[epoch]));
                sb.append(String.format(" (%.1f s)", (System.nanoTime() - t0) / 1e9));
                System.out.println(sb);
                System.out.flush();   // realtime progress, even when piped
            }

            if (earlyStop != null) {
                if (earlyStop.improved()) bestWeights = snapshot(params);
                stopped = earlyStop.update(metric);
            }
            done = epoch + 1;
        }

        if (stopped && bestWeights != null) restore(params, bestWeights);
        return trimHist(lossH, accH, vlH, vaH, done);
    }

    /** Inference-mode CE loss and accuracy over a labeled set: {loss, acc}. */
    public double[] evaluate(float[][] X, int[] Y, int classes) {
        float[][] logits = predict(X);
        double ce = 0;
        int correct = 0;
        for (int i = 0; i < X.length; i++) {
            int p = argmaxAt(logits[i], 0, classes);
            if (p == Y[i]) correct++;
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < classes; j++) max = Math.max(max, logits[i][j]);
            double sum = 0;
            for (int j = 0; j < classes; j++)
                sum += Math.exp(logits[i][j] - max);
            ce += -(float) Math.log(Math.max(
                    Math.exp(logits[i][Y[i]] - max) / sum, 1e-12));
        }
        return new double[]{ce / X.length, correct / (double) X.length};
    }

    private List<float[]> snapshot(List<Tensor> params) {
        List<float[]> snap = new ArrayList<>(params.size());
        for (Tensor p : params) snap.add(p.data.clone());
        return snap;
    }

    private static void restore(List<Tensor> params, List<float[]> snap) {
        for (int i = 0; i < params.size(); i++)
            System.arraycopy(snap.get(i), 0, params.get(i).data, 0,
                    snap.get(i).length);
    }

    private static History trimHist(double[] l, double[] a,
                                    double[] vl, double[] va, int n) {
        return new History(java.util.Arrays.copyOf(l, n),
                java.util.Arrays.copyOf(a, n),
                vl == null ? null : java.util.Arrays.copyOf(vl, n),
                va == null ? null : java.util.Arrays.copyOf(va, n));
    }

    // --------------------------- persistence -----------------------------

    /** Writes layer specs, parameters and buffers to a compact binary file. */
    public void saveTo(java.nio.file.Path path) throws java.io.IOException {
        try (var out = new java.io.DataOutputStream(
                new java.io.BufferedOutputStream(
                        java.nio.file.Files.newOutputStream(path)))) {
            out.writeInt(0x50544631);              // "PTF1"
            out.writeInt(firstInputDim);
            List<Layer> stack = layers;
            out.writeInt(stack.size());
            for (Layer l : stack) {
                out.writeInt(l.typeCode());
                int[] args = l.specArgs();
                out.writeInt(args.length);
                for (int a : args) out.writeInt(a);
            }
            for (Tensor p : params()) {
                out.writeInt(p.shape.length);
                for (int s : p.shape) out.writeInt(s);
                for (float v : p.data) out.writeFloat(v);
            }
            for (Layer l : stack) l.saveState(out);
        }
    }

    /** Rebuilds a trained network from {@link #saveTo} format. */
    public static Sequential loadFrom(java.nio.file.Path path)
            throws java.io.IOException {
        try (var in = new java.io.DataInputStream(
                new java.io.BufferedInputStream(
                        java.nio.file.Files.newInputStream(path)))) {
            if (in.readInt() != 0x50544631)
                throw new java.io.IOException("not a PTF1 model file");
            int inputDim = in.readInt();
            int k = in.readInt();
            Layer[] stack = new Layer[k];
            for (int i = 0; i < k; i++) {
                int code = in.readInt();
                int[] args = new int[in.readInt()];
                for (int j = 0; j < args.length; j++) args[j] = in.readInt();
                stack[i] = create(code, args);
            }
            Sequential net = new Sequential();
            for (Layer l : stack) net.add(l);

            // dummy forward triggers lazy init so params exist
            net.forward(Tensor.of(new float[inputDim], 1, inputDim));
            List<Tensor> ps = net.params();
            for (Tensor p : ps) {
                int rank = in.readInt();
                int[] shape = new int[rank];
                int len = 1;
                for (int j = 0; j < rank; j++) {
                    shape[j] = in.readInt();
                    len *= shape[j];
                }
                if (len != p.data.length)
                    throw new java.io.IOException("param size mismatch on load");
                for (int j = 0; j < len; j++) p.data[j] = in.readFloat();
            }
            for (Layer l : stack) l.loadState(in);
            return net;
        }
    }

    private static Layer create(int code, int[] args) {
        return switch (code) {
            case 1 -> new Dense(args[0], Dense.Act.values()[args[1]]);
            case 2 -> new Dropout(args[0] / 1000f);
            case 3 -> new Flatten();
            case 4 -> new ReLU();
            case 5 -> new LeakyReLU(args[0] / 10000f);
            case 6 -> new ELU(args[0] / 1000f);
            case 7 -> new SELU();
            case 8 -> new SiLU();
            case 9 -> new GELU();
            case 10 -> new Softplus();
            case 11 -> new Sigmoid();
            case 12 -> new Tanh();
            case 13 -> new Softmax();
            case 14 -> new Reshape(args);
            case 15 -> new GlobalAveragePooling();
            case 16 -> new LayerNorm1D(args.length > 0 ? args[0] / 1e7f : 1e-5f);
            case 17 -> new BatchNorm1D(
                    args.length > 0 ? args[0] / 10000f : 0.99f,
                    args.length > 1 ? args[1] / 1e7f : 1e-5f);
            case 18 -> new Embedding(args[0], args[1]);
            case 19 -> new PositionalEncoding();
            case 20 -> new SelfAttention(args.length > 0 && args[0] == 1);
            case 21 -> new MultiHeadAttention(args[0], args.length > 1 && args[1] == 1);
            case 22 -> new ZeroPad2D(args[0], args[1], args[2], args[3]);
            case 23 -> new RNN(args[0], args.length > 1 && args[1] == 1);
            case 24 -> new LSTM(args[0], args.length > 1 && args[1] == 1);
            case 25 -> new GRU(args[0], args.length > 1 && args[1] == 1);
            case 26 -> new Conv2D(args[0], args[1], args[2], args[3], args[4],
                    Padding.of(args[5]));
            case 28 -> new MaxPool2D(args[0], args[1], args[2], args[3],
                    Padding.of(args[4]));
            case 29 -> new AvgPool2D(args[0], args[1], args[2], args[3],
                    Padding.of(args[4]));
            case 30 -> new Upsample2D(args[0], args[1]);
            case 31 -> new BatchNorm2D(
                    args.length > 0 ? args[0] / 10000f : 0.99f,
                    args.length > 1 ? args[1] / 1e7f : 1e-5f);
            default -> throw new IllegalArgumentException(
                    "unknown layer code " + code);
        };
    }

    /** Argmax over {@code a[offset .. offset + k)} - one row of logits. */
    private static int argmaxAt(float[] a, int offset, int k) {
        int best = 0;
        for (int j = 1; j < k; j++) if (a[offset + j] > a[offset + best]) best = j;
        return best;
    }

    /** Raw outputs of the final layer (logits for classifiers). */
    public float[][] predict(float[][] X) {
        setTraining(false);
        Tensor.resetGraph();
        int n = X.length, d = X[0].length;
        Tensor out = forward(Tensor.of(flatten(X), n, d));
        int cols = out.data.length / n;
        float[][] res = new float[n][cols];
        for (int i = 0; i < n; i++)
            System.arraycopy(out.data, i * cols, res[i], 0, cols);
        return res;
    }

    /** Argmax class ids from {@link #predict}. */
    public int[] predictClass(float[][] X) {
        float[][] out = predict(X);
        int[] res = new int[out.length];
        for (int i = 0; i < out.length; i++) res[i] = argmaxAt(out[i], 0, out[i].length);
        return res;
    }

    /** Predict a single sample. Returns the final layer outputs (logits). */
    public float[] predict(float[] x) {
        setTraining(false);
        Tensor.resetGraph();
        Tensor out = forward(Tensor.of(x, 1, x.length));
        float[] res = new float[out.data.length];
        System.arraycopy(out.data, 0, res, 0, res.length);
        return res;
    }

    /** Class probabilities (softmax of logits) for one sample. */
    public float[] predictProb(float[] x) {
        return softmaxRow(predict(x), -1);
    }

    /** Class probabilities for every sample: float[n][classes]. */
    public float[][] predictProb(float[][] X) {
        float[][] logits = predict(X);
        for (float[] r : logits) softmaxRow(r, -1);
        return logits;
    }

    /**
     * Indices of the top {@code k} predicted classes per sample. Rows are the
     * logits from {@link #predict}.
     */
    public int[][] predictTopK(float[][] X, int k) {
        float[][] logits = predict(X);
        int[][] res = new int[logits.length][];
        for (int i = 0; i < logits.length; i++) res[i] = topK(logits[i], k);
        return res;
    }

    /**
     * All trainable weights, flattened, in layer order: one float[] per
     * parameter tensor. Shape is preserved by {@link #setWeights}. Useful for
     * inspection, snapshotting, or weight surgery.
     */
    public java.util.List<float[]> getWeights() {
        var out = new java.util.ArrayList<float[]>();
        for (Tensor p : params()) out.add(p.data.clone());
        return out;
    }

    /**
     * Copies flattened weights (as returned by {@link #getWeights}) into the
     * model's existing parameter tensors. Total element count must match.
     */
    public void setWeights(java.util.List<float[]> ws) {
        java.util.List<Tensor> ps = params();
        if (ps.size() != ws.size())
            throw new IllegalArgumentException("weight count mismatch");
        for (int i = 0; i < ps.size(); i++) {
            float[] src = ws.get(i);
            float[] dst = ps.get(i).data;
            if (src.length != dst.length)
                throw new IllegalArgumentException("param size mismatch at " + i);
            System.arraycopy(src, 0, dst, 0, dst.length);
        }
    }

    /**
     * Deep copy of the model: a new {@code Sequential} with the same layer
     * architecture, the same weights and any extra layer state (e.g. BatchNorm
     * running statistics). The copy is fully independent of this instance.
     *
     * <p>Useful for K-fold, warm restarts, or mutating one branch of an
     * experiment without disturbing the original. Requires all layers to be
     * persistable (see {@link Layer#typeCode}).</p>
     *
     * @throws IllegalStateException if a layer is not persistable or the model
     *                               was built without a recorded input dim
     */
    public Sequential copy() {
        Sequential c = new Sequential();
        for (Layer l : layers) {
            int code = l.typeCode();
            if (code < 0)
                throw new IllegalStateException(
                        "cannot copy non-persistable layer " + l.typeName());
            c.add(create(code, l.specArgs()));
        }
        c.firstInputDim = firstInputDim;
        if (!layers.isEmpty() && layers.get(0).initialized()) {
            if (firstInputDim <= 0)
                throw new IllegalStateException("input dim not recorded; cannot copy weights");
            c.build(firstInputDim);
            c.setWeights(getWeights());
            try {
                var bo = new java.io.ByteArrayOutputStream();
                var dOut = new java.io.DataOutputStream(bo);
                for (Layer l : layers) l.saveState(dOut);
                dOut.flush();
                var dIn = new java.io.DataInputStream(
                        new java.io.ByteArrayInputStream(bo.toByteArray()));
                for (Layer cl : c.layers) cl.loadState(dIn);
            } catch (java.io.IOException e) {
                throw new IllegalStateException("copy failed while cloning state", e);
            }
        }
        return c;
    }

    /**
     * Global gradient-norm clipping across every trainable parameter: if the
     * summed L2 norm of all gradients exceeds {@code maxNorm}, all gradients
     * are scaled down proportionally. Call after {@code loss.backward()} and
     * before the optimizer {@code step}. Helps stabilize RNN/GAN training.
     * Returns this.
     */
    public Sequential clipGradNorm(float maxNorm) {
        List<Tensor> ps = params();
        double sum = 0;
        for (Tensor p : ps) {
            if (p.grad == null) continue;
            for (float g : p.grad) sum += (double) g * g;
        }
        double norm = Math.sqrt(sum);
        if (norm > maxNorm && norm > 0) {
            float k = maxNorm / (float) norm;
            for (Tensor p : ps) {
                if (p.grad == null) continue;
                for (int i = 0; i < p.grad.length; i++) p.grad[i] *= k;
            }
        }
        return this;
    }

    // ---------------- format-agnostic interchange hub --------------------

    /**
     * Snapshots this trained model into a {@link puretf.serialize.ModelBundle},
     * the common representation used by the safetensors / ONNX / PyTorch /
     * TensorFlow import-export drivers. Every parameter tensor is stored under
     * a readable canonical name (e.g. {@code layer_2/kernel},
     * {@code layer_4/recurrent}) and the layer architecture is recorded as a
     * manifest so the model can be rebuilt exactly. The model's recorded input
     * dimension is saved in the bundle metadata.
     */
    public puretf.serialize.ModelBundle saveBundle() {
        puretf.serialize.ModelBundle b = new puretf.serialize.ModelBundle();
        for (int li = 0; li < layers.size(); li++) {
            Layer l = layers.get(li);
            b.recordLayer(l.typeName(), l.typeCode(), l.specArgs());
            java.util.List<Tensor> ps = l.params();
            for (int k = 0; k < ps.size(); k++) {
                Tensor p = ps.get(k);
                String role = roleFor(l.typeName(), k);
                String name = "layer_" + li + "/" + role;
                b.add(puretf.serialize.NamedTensor.fromFloats(name, p.shape, p.data));
            }
        }
        b.metadata().put("inputDim", Integer.toString(firstInputDim));
        b.metadata().put("format", "pure-tf-java");
        return b;
    }

    /**
     * Rebuilds a new {@code Sequential} from a bundle produced by
     * {@link #saveBundle} or by deserializing an external format. The layer
     * manifest reconstructs the architecture and the named tensors restore the
     * weights (matched by layer index + parameter index).
     */
    public static Sequential loadBundle(puretf.serialize.ModelBundle b) {
        Sequential net = new Sequential();
        var manifest = b.layerManifest();
        for (var spec : manifest) net.add(createLayer(spec.code, spec.args));

        int inputDim = Integer.parseInt(
                b.metadata().getOrDefault("inputDim", "-1"));
        if (inputDim <= 0 && !manifest.isEmpty()) {
            // bellwether: fan-in of the first real layer where available
            var first = b.paramByIndex(0);
            int d = first == null ? 1 : first.shape()[first.rank() - 1];
            inputDim = d > 0 ? d : 1;
        }
        List<Tensor> ps;
        if (inputDim > 0) {
            net.forward(Tensor.of(new float[inputDim], 1, inputDim));
            ps = net.params();
        } else {
            throw new IllegalArgumentException(
                    "bundle has no inputDim metadata and cannot size the model");
        }

        // weights are stored in exact layer/param order; copy them across
        int idx = 0;
        for (int li = 0; li < net.layers.size(); li++) {
            java.util.List<Tensor> lps = net.layers.get(li).params();
            for (int k = 0; k < lps.size(); k++, idx++) {
                if (idx >= b.size())
                    throw new IllegalArgumentException("bundle has fewer tensors than the model needs");
                puretf.serialize.NamedTensor t = b.paramByIndex(idx);
                float[] f = t.toFloats();
                if (f.length != lps.get(k).data.length)
                    throw new IllegalArgumentException(
                            "param size mismatch at layer " + li + " param " + k
                                    + ": bundle " + f.length + " vs model " + lps.get(k).data.length);
                System.arraycopy(f, 0, lps.get(k).data, 0, f.length);
            }
        }
        return net;
    }

    // ---- name helpers ----

    private static String roleFor(String type, int k) {
        switch (type) {
            case "Dense":
            case "Conv2D": return k == 0 ? "kernel" : "bias";
            case "Embedding": return "table";
            case "LSTM":
            case "RNN":
            case "GRU": return k == 0 ? "kernel" : k == 1 ? "recurrent" : "bias";
            case "BatchNorm1D":
            case "BatchNorm2D":
            case "LayerNorm1D": return k == 0 ? "gamma" : "beta";
            case "MultiHeadAttention": return k == 0 ? "q" : k == 1 ? "k" : k == 2 ? "v" : "o";
            default: return "param" + k;
        }
    }

    /** Shallow wrapper so {@link #loadBundle} can reuse {@link #create}. */
    private static Layer createLayer(int code, int[] args) {
        return create(code, args);
    }

    // --------------------- row helpers ------------------------------------

    /** In-place softmax of {@code a} if {@code len < 0}, else first {@code len}; returns a. */
    private static float[] softmaxRow(float[] a, int ignore) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : a) max = Math.max(max, v);
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            a[i] = (float) Math.exp(a[i] - max);
            sum += a[i];
        }
        for (int i = 0; i < a.length; i++) a[i] /= (float) sum;
        return a;
    }

    /** Indices of the {@code k} largest values in {@code a}, descending. */
    private static int[] topK(float[] a, int k) {
        int kk = Math.min(Math.max(k, 1), a.length);
        int[] idx = new int[a.length];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        // selection sort, keep top-k only
        for (int i = 0; i < kk; i++) {
            int best = i;
            for (int j = i + 1; j < a.length; j++)
                if (a[j] > a[best]) best = j;
            int t = idx[i]; idx[i] = idx[best]; idx[best] = t;
            float f = a[i]; a[i] = a[best]; a[best] = f;
        }
        int[] out = new int[kk];
        System.arraycopy(idx, 0, out, 0, kk);
        return out;
    }

    private static float[] flatten(float[][] m) {
        float[] out = new float[m.length * m[0].length];
        for (int i = 0; i < m.length; i++)
            System.arraycopy(m[i], 0, out, i * m[0].length, m[0].length);
        return out;
    }

    private static void shuffle(int[] a, Random rnd) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = a[i]; a[i] = a[j]; a[j] = t;
        }
    }
}
