package puretf.nn;

import puretf.Tensor;

import java.util.List;

/**
 * Batch normalization over the batch axis for rank-2 input [B,F] with
 * learned affine gain/bias and frozen running stats at inference
 * (Ioffe & Szegedy 2015). Training path is composed from primitive autograd
 * ops; running mean/var update after each training forward.
 */
public final class BatchNorm1D extends Layer {

    private final float eps, momentum;
    private Tensor gamma, beta;
    private float[] runningMean, runningVar;
    private boolean training = true;

    public BatchNorm1D() { this(0.99f, 1e-5f); }

    public BatchNorm1D(float momentum, float eps) {
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
        int f = fanIn();
        if (!training) {
            float[] out = new float[x.data.length];
            for (int i = 0; i < x.data.length; i++) {
                int j = i % f;
                out[i] = (x.data[i] - runningMean[j])
                        / (float) Math.sqrt(runningVar[j] + eps)
                        * gamma.data[j] + beta.data[j];
            }
            return Tensor.of(out, x.shape.clone());
        }
        Tensor mu = x.meanAxis(0);
        Tensor xc = x.broadcastAdd(mu.scale(-1f));
        Tensor var = xc.mul(xc).meanAxis(0);
        Tensor invStd = var.addScalar(eps).power(-0.5f);
        Tensor xhat = xc.broadcastMul(invStd);
        updateRunning(x.data, x.shape[0], f);
        return xhat.broadcastMul(gamma).broadcastAdd(beta);
    }

    private void updateRunning(float[] data, int b, int f) {
        float[] mean = new float[f], m2 = new float[f];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < f; j++) {
                float v = data[i * f + j];
                mean[j] += v;
                m2[j] += v * v;
            }
        for (int j = 0; j < f; j++) {
            mean[j] /= b;
            float var = m2[j] / b - mean[j] * mean[j];
            runningMean[j] = momentum * runningMean[j] + (1 - momentum) * mean[j];
            runningVar[j] = momentum * runningVar[j] + (1 - momentum) * var;
        }
    }

    @Override
    public List<Tensor> params() { return List.of(gamma, beta); }

    @Override public int typeCode() { return 17; }

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
        in.readInt();   // fanIn sanity check omitted; sizes are fixed by init
        for (int i = 0; i < runningMean.length; i++) runningMean[i] = in.readFloat();
        for (int i = 0; i < runningVar.length; i++) runningVar[i] = in.readFloat();
    }
}
