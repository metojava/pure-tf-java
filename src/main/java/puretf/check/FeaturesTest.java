package puretf.check;

import puretf.Optim;
import puretf.nn.*;
import puretf.nn.EarlyStopping.Monitor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/**
 * Integration smoke test for the ported stack: LayerNorm/BatchNorm/GELU/
 * Embedding-style composition, Keras-style trainer (validation split,
 * cosine schedule, early stopping) and model save/load round-trip.
 * Run: run.cmd features
 */
public final class FeaturesTest {

    public static void main(String[] args) throws Exception {
        // ---------------- toy 3-class spiral dataset ---------------------
        int n = 600;
        float[][] X = new float[n][2];
        int[] Y = new int[n];
        Random rnd = new Random(1L);
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < n / 3; i++) {
                int k = c * (n / 3) + i;
                double t = i / (double) (n / 3) * 3.5 + rnd.nextGaussian() * 0.12;
                X[k][0] = (float) (t * Math.sin(t + c * 2 * Math.PI / 3));
                X[k][1] = (float) (t * Math.cos(t + c * 2 * Math.PI / 3));
                Y[k] = c;
            }

        Sequential net = new Sequential()
                .add(new Dense(64))
                .add(new LayerNorm1D())
                .add(new GELU())
                .add(new Dense(32))
                .add(new BatchNorm1D())
                .add(new SiLU())
                .add(new Dropout(0.1f))
                .add(new Dense(3));

        History h = net.fitHistory(X, Y, 3, 60, 64,
                new Optim.Adam(0.005f),
                LrSchedule.cosine(0.005f, 0.0005f, 60),
                0.2f,
                new EarlyStopping(Monitor.LOSS, 100, 0.0),
                true);
        double finalValAcc = h.valAcc()[h.valAcc().length - 1];
        System.out.printf("%nfinal val acc %.4f%n", finalValAcc);
        if (finalValAcc < 0.90)
            throw new AssertionError("val accuracy too low: " + finalValAcc);

        // ---------------- save / load round-trip -------------------------
        Path tmp = Files.createTempFile("puretf-smoke", ".bin");
        net.saveTo(tmp);
        Sequential restored = Sequential.loadFrom(tmp);
        float[][] p1 = net.predict(X);
        float[][] p2 = restored.predict(X);
        double drift = 0;
        for (int i = 0; i < p1.length; i++)
            for (int j = 0; j < p1[0].length; j++)
                drift = Math.max(drift, Math.abs(p1[i][j] - p2[i][j]));
        System.out.printf("save/load max logit drift %.8f (%s)%n", drift, tmp);
        if (drift > 1e-4f)
            throw new AssertionError("restored model diverges, drift=" + drift);

        System.out.println("\nfeatures smoke test passed");

        // ================= sequence stack =================================
        // task: report the SECOND token of each length-6 id sequence
        // (requires positional awareness -> exercises PosEnc/attention/LSTM)
        int seqN = 900, T = 6, vocab = 5;
        float[][] SX = new float[seqN][T];
        int[] SY = new int[seqN];
        Random srnd = new Random(9L);
        for (int i = 0; i < seqN; i++) {
            for (int j = 0; j < T; j++) SX[i][j] = srnd.nextInt(vocab);
            SY[i] = (int) SX[i][1];
        }

        Sequential att = new Sequential()
                .add(new Embedding(vocab, 16))
                .add(new PositionalEncoding())
                .add(new MultiHeadAttention(2))
                .add(new SelfAttention())
                .add(new GlobalAveragePooling())
                .add(new Dense(32, Dense.Act.RELU))
                .add(new Dense(vocab));
        History ah = att.fitHistory(SX, SY, vocab, 40, 64,
                new Optim.Adam(0.005f), LrSchedule.constant(0.005f),
                0.2f, null, true);
        double attAcc = ah.valAcc()[ah.valAcc().length - 1];
        System.out.printf("%nattention final val acc %.4f%n", attAcc);
        if (attAcc < 0.90) throw new AssertionError("attention acc: " + attAcc);

        Sequential rnn = new Sequential()
                .add(new Embedding(vocab, 12))
                .add(new LSTM(24))
                .add(new Dense(vocab));
        History rh = rnn.fitHistory(SX, SY, vocab, 40, 64,
                new Optim.Adam(0.005f), LrSchedule.constant(0.005f),
                0.2f, null, true);
        double rnnAcc = rh.valAcc()[rh.valAcc().length - 1];
        System.out.printf("lstm final val acc %.4f%n", rnnAcc);
        if (rnnAcc < 0.90) throw new AssertionError("lstm acc: " + rnnAcc);

        // save/load round-trip including the sequence layers
        Path tmp2 = Files.createTempFile("puretf-seq", ".bin");
        att.saveTo(tmp2);
        Sequential attRestored = Sequential.loadFrom(tmp2);
        float[][] q1 = att.predict(SX);
        float[][] q2 = attRestored.predict(SX);
        double drift2 = 0;
        for (int i = 0; i < q1.length; i++)
            for (int j = 0; j < q1[0].length; j++)
                drift2 = Math.max(drift2, Math.abs(q1[i][j] - q2[i][j]));
        if (drift2 > 1e-4f)
            throw new AssertionError("restored sequence model diverges");
        System.out.println("sequence smoke test passed");

        // ================= vision stack ===================================
        // task: is the bright 3x3 patch in the LEFT or RIGHT half of an 8x8
        // image? (requires spatial awareness -> exercises conv + pool)
        int imgN = 500, S = 8;
        float[][] IX = new float[imgN][S * S];
        int[] IY = new int[imgN];
        Random irnd = new Random(5L);
        for (int i = 0; i < imgN; i++) {
            boolean left = i % 2 == 0;
            IY[i] = left ? 0 : 1;
            int py = irnd.nextInt(S - 3), px = left
                    ? irnd.nextInt(S / 2 - 3)
                    : S / 2 + irnd.nextInt(S / 2 - 3);
            for (int y = 0; y < S; y++)
                for (int x = 0; x < S; x++) {
                    boolean inPatch = y >= py && y < py + 3 && x >= px && x < px + 3;
                    IX[i][y * S + x] = inPatch ? 0.9f
                            : irnd.nextFloat() * 0.15f;
                }
        }

        Sequential cnn = new Sequential()
                .add(new Reshape(-1, S, S, 1))
                .add(new Conv2D(4, 3, 3, 1, 1, Padding.SAME))
                .add(new ReLU())
                .add(new MaxPool2D(2, 2, 2, 2, Padding.VALID))
                .add(new Flatten())
                .add(new Dense(16, Dense.Act.RELU))
                .add(new Dense(2));
        History ch2 = cnn.fitHistory(IX, IY, 2, 12, 32,
                new Optim.Adam(0.003f), LrSchedule.constant(0.003f),
                0.2f, null, true);
        double cnnAcc = ch2.valAcc()[ch2.valAcc().length - 1];
        System.out.printf("%ncnn final val acc %.4f%n", cnnAcc);
        if (cnnAcc < 0.95) throw new AssertionError("cnn acc: " + cnnAcc);

        Path tmp3 = Files.createTempFile("puretf-cnn", ".bin");
        cnn.saveTo(tmp3);
        Sequential cnnRestored = Sequential.loadFrom(tmp3);
        float[][] r1 = cnn.predict(IX);
        float[][] r2 = cnnRestored.predict(IX);
        double drift3 = 0;
        for (int i = 0; i < r1.length; i++)
            for (int j = 0; j < r1[0].length; j++)
                drift3 = Math.max(drift3, Math.abs(r1[i][j] - r2[i][j]));
        if (drift3 > 1e-4f)
            throw new AssertionError("restored cnn diverges: " + drift3);

        System.out.println("vision smoke test passed");
    }

    private FeaturesTest() {}
}
