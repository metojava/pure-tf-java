package puretf.demo;

import puretf.Optim;
import puretf.Tensor;
import puretf.nn.Dense;
import puretf.nn.LeakyReLU;
import puretf.nn.Sequential;

import java.util.Random;

/**
 * Minimal Generative Adversarial Network on 2D data, pure Java, no external
 * deps. The generator maps latent noise to fake points; the discriminator
 * tries to tell real from fake. Both train with the numerically-stable
 * {@link Tensor#bce} loss in an alternating adversarial loop,
 * demonstrating the manual-gradient training pattern the library exposes
 * for GAN-style / sequence models.
 *
 * <p>Target: four well-separated 2D Gaussians. After training, the generator
 * should place samples near those four clusters - we measure that by counting
 * how many generated points fall within a small radius of a cluster center.</p>
 */
public final class GanDemo {

    // target clusters, shrunk into tanh's [-1,1] output range
    private static final float[][] CENTERS = {
            {-0.6f, -0.6f}, {0.6f, -0.6f}, {-0.6f, 0.6f}, {0.6f, 0.6f}};
    private static final float CLUSTER_SD = 0.08f;

    public static void main(String[] args) throws Exception {
        int batch = 64;
        int latent = 2;
        int iters = 6000;
        Random rnd = new Random(7L);

        Sequential G = new Sequential()
                .add(new Dense(32, Dense.Act.RELU))
                .add(new Dense(32, Dense.Act.RELU))
                .add(new Dense(2, Dense.Act.TANH));          // -> x in [-1,1]
        G.build(latent);

        Sequential D = new Sequential()
                .add(new Dense(32, Dense.Act.NONE))
                .add(new LeakyReLU(0.2f))
                .add(new Dense(32, Dense.Act.NONE))
                .add(new LeakyReLU(0.2f))
                .add(new Dense(1));                          // logit, then BCE
        D.build(2);

        Optim.Adam gOpt = new Optim.Adam(0.001f);
        Optim.Adam dOpt = new Optim.Adam(0.001f);

        Tensor ones  = Tensor.of(ones(batch), batch, 1);
        Tensor zeros = Tensor.of(zeros0(batch), batch, 1);

        for (int it = 0; it < iters; it++) {
            // --- sample real + fake --------------------------------------
            float[][] real = realBatch(rnd, batch);
            Tensor xReal = Tensor.of(flatten(real), batch, 2);
            Tensor z = Tensor.randn(rnd, 1f, batch, latent);

            // --- train discriminator on real + fake ----------------------
            Tensor.resetGraph();
            Tensor gOut = G.forward(z);
            Tensor realLogit = D.forward(xReal);
            Tensor fakeLogit = D.forward(gOut);
            Tensor dLoss = realLogit.bce(ones).broadcastAdd(fakeLogit.bce(zeros));
            for (Tensor p : D.params()) p.zeroGrad();
            dLoss.backward();
            D.clipGradNorm(1f);
            dOpt.step(D.params());
            float dErr = dLoss.data[0];

            // --- train generator to fool discriminator -------------------
            Tensor.resetGraph();
            z = Tensor.randn(rnd, 1f, batch, latent);
            gOut = G.forward(z);
            fakeLogit = D.forward(gOut);
            Tensor gLoss = fakeLogit.bce(ones);              // want D to say "real"
            for (Tensor p : G.params()) p.zeroGrad();
            gLoss.backward();
            G.clipGradNorm(1f);
            gOpt.step(G.params());
            float gErr = gLoss.data[0];

            if (it % 500 == 0 || it == iters - 1)
                System.out.printf("iter %6d  D loss %.4f  G loss %.4f%n",
                        it, dErr, gErr);
        }

        // ---- score: fraction of generated points near a cluster center ---
        int n = 400, hit = 0;
        float[][] gen = new float[n][2];
        for (int i = 0; i < n; i++) {
            Tensor.resetGraph();
            float[] p = G.predict(Tensor.randn(rnd, 1f, 1, latent).data);
            gen[i][0] = p[0]; gen[i][1] = p[1];
            if (nearCluster(p)) hit++;
        }
        System.out.printf("generated points landing near a real cluster: %d/%d (%.1f%%)%n",
                hit, n, 100.0 * hit / n);
        System.out.println("(perfect GAN on 4 well-separated 2D clusters approaches ~100%)");
    }

    private static boolean nearCluster(float[] p) {
        for (float[] c : CENTERS) {
            float dx = p[0] - c[0], dy = p[1] - c[1];
            if (dx * dx + dy * dy < 0.02f) return true;   // radius ~0.14
        }
        return false;
    }

    private static float[][] realBatch(Random rnd, int batch) {
        float[][] out = new float[batch][2];
        for (int i = 0; i < batch; i++) {
            float[] c = CENTERS[rnd.nextInt(CENTERS.length)];
            out[i][0] = c[0] + (float) (rnd.nextGaussian() * CLUSTER_SD);
            out[i][1] = c[1] + (float) (rnd.nextGaussian() * CLUSTER_SD);
        }
        return out;
    }

    private static float[] ones(int n) {
        float[] o = new float[n];
        java.util.Arrays.fill(o, 1f);
        return o;
    }

    private static float[] zeros0(int n) { return new float[n]; }

    private static float[] flatten(float[][] m) {
        float[] o = new float[m.length * m[0].length];
        for (int i = 0; i < m.length; i++)
            System.arraycopy(m[i], 0, o, i * m[0].length, m[0].length);
        return o;
    }
}
