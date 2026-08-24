package puretf.nn;

import puretf.Tensor;

import java.util.List;

/**
 * Batch normalization for NHWC tensors: statistics over batch and spatial
 * axes per channel. Training path composed from primitive autograd ops;
 * frozen running stats at inference.
 */
public final class BatchNorm2D extends Layer {

    private final float eps, momentum;
    private Tensor gamma, beta;
    private float[] runningMean, runningVar;
    private boolean training = true;

    public BatchNorm2D() { this(0.99f, 1e-5f); }

    public BatchNorm2D(float momentum, float eps) {
        this.momentum = momentum;
        this.eps = eps;
    }

    @Override
    protected void init(int fanIn) {
        float[] ones = new float[fanIn];
        java.util.Arrays.fill(ones, 1f);
        gamma = Tensor.param(ones, fanIn);
        beta = Tensor.param(new float[fanIn], fanIn);
        runningMean = new float[fanIn];
        runningVar = new float[fanIn];
        java.util.Arrays.fill(runningVar, 1f);
    }

    @Override
    public void setTraining(boolean training) { this.training = training; }

    @Override
    protected Tensor apply(Tensor x) {
        int c = fanIn();
        if (!training) {
            float[] out = new float[x.data.length];
            for (int i = 0; i < x.data.length; i++) {
                int j = i % c;
                out[i] = (x.data[i] - runningMean[j])
                        / (float) Math.sqrt(runningVar[j] + eps)
                        * gamma.data[j] + beta.data[j];
            }
            return Tensor.of(out, x.shape.clone());
        }
        // mean/var over batch+spatial axes -> [C]
        Tensor mu = x.meanAxis(0).meanAxis(1).meanAxis(1);
        Tensor xc = x.broadcastAdd(mu.scale(-1f));
        Tensor var = xc.mul(xc).meanAxis(0).meanAxis(1).meanAxis(1);
        Tensor invStd = var.addScalar(eps).power(-0.5f);
        updateRunning(x.data, x.data.length / c, c);
        return xc.broadcastMul(invStd).broadcastMul(gamma)
                .broadcastAdd(beta);
    }

    private void updateRunning(float[] data, int n, int c) {
        float[] mean = new float[c], m2 = new float[c];
        for (int i = 0; i < data.length; i++) {
            int j = i % c;
            mean[j] += data[i];
            m2[j] += data[i] * data[i];
        }
        for (int j = 0; j < c; j++) {
            mean[j] /= n;
            float var = m2[j] / n - mean[j] * mean[j];
            runningMean[j] = momentum * runningMean[j] + (1 - momentum) * mean[j];
            runningVar[j] = momentum * runningVar[j] + (1 - momentum) * var;
        }
    }

    @Override
    public List<Tensor> params() { return List.of(gamma, beta); }

    @Override public int typeCode() { return 31; }

    @Override public int[] specArgs() {
        return new int[]{Math.round(momentum * 10000), Math.round(eps * 1e7f)};
    }

    @Override
    public void saveState(java.io.DataOutputStream out) throws java.io.IOException {
        out.writeInt(fanIn());
        for (float v : runningMean) out.writeFloat(v);
        for (float v : runningVar) out.writeFloat(v);
    }

    @Override
    public void loadState(java.io.DataInputStream in) throws java.io.IOException {
        in.readInt();
        for (int i = 0; i < runningMean.length; i++) runningMean[i] = in.readFloat();
        for (int i = 0; i < runningVar.length; i++) runningVar[i] = in.readFloat();
    }
}
