package puretf.data;

import java.util.Random;

/**
 * Categorical sampling helpers used by generative and text models: draw an
 * index from a probability vector, optionally with a temperature scale (for
 * {code LSTM} character generation) or simple argmax (greedy decoding).
 */
public final class Sampler {

    private Sampler() {}

    /**
     * Draws an index from a discrete distribution given by {@code logits}
     * (any real values; softmax applied internally). Additively scales by
     * {@code temperature} before normalizing: higher = more random, lower =
     * more peaky. A temperature of {@code 0} degenerates to greedy argmax.
     */
    public static int categorical(Random rnd, float[] logits, float temperature) {
        int n = logits.length;
        if (temperature <= 0f) return argmax(logits);
        double[] p = new double[n];
        double t = temperature;
        double max = Double.NEGATIVE_INFINITY;
        for (float v : logits) max = Math.max(max, v / t);
        double sum = 0;
        for (int i = 0; i < n; i++) {
            p[i] = Math.exp(logits[i] / t - max);
            sum += p[i];
        }
        double u = rnd.nextDouble() * sum;
        for (int i = 0; i < n; i++) {
            u -= p[i];
            if (u <= 0) return i;
        }
        return n - 1;
    }

    /** Greedy index of the largest value. */
    public static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
        return best;
    }
}
