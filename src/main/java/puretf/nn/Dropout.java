package puretf.nn;

import puretf.Tensor;

import java.util.List;
import java.util.Random;

/** Inverted dropout: active only while {@code setTraining(true)}. */
public final class Dropout extends Layer {

    private final float rate;
    private boolean training = false;
    private float[] mask;
    private int lastFanIn = -1;

    public Dropout(float rate) {
        if (rate < 0 || rate >= 1) throw new IllegalArgumentException("rate");
        this.rate = rate;
        this.seed = seedCounter++;
    }

    private final long seed;
    private static long seedCounter = 0;

    @Override
    public void setTraining(boolean training) { this.training = training; }

    @Override
    public List<Tensor> params() { return List.of(); }

    @Override
    protected void init(int fanIn) { lastFanIn = fanIn; }

    @Override
    protected Tensor apply(Tensor x) {
        if (!training) return x;
        int n = x.data.length;
        mask = new float[n];
        Random rnd = new Random(seed ^ System.nanoTime());
        float keep = 1f - rate;
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            mask[i] = rnd.nextFloat() < keep ? 1f / keep : 0f;
            out[i] = x.data[i] * mask[i];
        }
        Tensor r = Tensor.opPublic(out, x.shape.clone(), x);
        r.setBackprop(() -> {
            float[] g = new float[n];
            for (int i = 0; i < n; i++) g[i] = r.grad[i] * mask[i];
            Tensor.accPublic(x, g);
        });
        return r;
    }

    @Override public int typeCode() { return 2; }

    @Override public int[] specArgs() { return new int[]{Math.round(rate * 1000)}; }
}
