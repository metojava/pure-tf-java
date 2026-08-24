package puretf.demo;

import puretf.Tensor;
import puretf.data.IdxDataset;
import puretf.nn.Dense;
import puretf.nn.Dropout;
import puretf.nn.History;
import puretf.nn.Metrics;
import puretf.nn.Sequential;

/**
 * Fashion-MNIST classifier - the first image-scale workload for the
 * pure-Java engine. MLP: 784 -> 256 (tanh) -> dropout -> 10 logits,
 * trained with fused softmax cross-entropy + Adam.
 *
 * Usage: main [dataDir] [epochs]   (defaults: Desktop data dir, 20 epochs)
 */
public final class FashionMnistDemo {

    private static final String[] NAMES = {
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    public static void main(String[] args) throws Exception {
        String dir = "C:\\Users\\dell\\Desktop\\data\\fashion";
        int epochs = 5;
        int batch = 128;
        boolean showChart = false;
        for (String a : args) {
            if (a.matches("\\d+")) epochs = Integer.parseInt(a);
            else if (a.equals("--chart")) showChart = true;
            else if (a.startsWith("--batch=")) batch = Integer.parseInt(a.substring(8));
            else dir = a;
        }

        long t0 = System.currentTimeMillis();
        var trI = IdxDataset.loadImages(dir + "\\train-images-idx3-ubyte.gz");
        var trL = IdxDataset.loadLabels(dir + "\\train-labels-idx1-ubyte.gz");
        var teI = IdxDataset.loadImages(dir + "\\t10k-images-idx3-ubyte.gz");
        var teL = IdxDataset.loadLabels(dir + "\\t10k-labels-idx1-ubyte.gz");
        System.out.printf("loaded %d train / %d test %dx%d images (%.1fs)%n",
                trI.count(), teI.count(), trI.rows(), trI.cols(),
                (System.currentTimeMillis() - t0) / 1000.0);

        Sequential net = new Sequential()
                .add(new Dense(256, Dense.Act.TANH))
                .add(new Dropout(0.15f))
                .add(new Dense(10));

        long trainStart = System.currentTimeMillis();
        History hist = net.fitClassifier(trI.pixels(), trL.labels(), 10,
                epochs, batch, 0.001f, true);
        double trainSec = (System.currentTimeMillis() - trainStart) / 1000.0;
        System.out.printf("final epoch loss %.4f  |  training time %.1f s (%.1f s/epoch)%n",
                hist.loss()[hist.loss().length - 1], trainSec,
                trainSec / epochs);

        // ---------------- evaluate on the 10k test set -----------------
        float[][] logits = net.predict(teI.pixels());
        int correct = 0;
        int[][] conf = new int[10][10];
        double ceLoss = 0;
        int[] pred = new int[teL.count()];
        for (int i = 0; i < teL.count(); i++) {
            pred[i] = argmax(logits[i]);
            conf[teL.labels()[i]][pred[i]]++;
            if (pred[i] == teL.labels()[i]) correct++;
            ceLoss += -Math.log(Math.max(
                    softmax(logits[i])[teL.labels()[i]], 1e-12));
        }
        double acc = correct / (double) teL.count();
        Metrics.printReport(ceLoss / teL.count(), acc, conf, NAMES);
        Metrics.printDl4jMatrix(conf);

        // per-class accuracy + first test sample of every class
        double[] classAccPct = new double[10];
        int[] firstIdx = new int[10];
        java.util.Arrays.fill(firstIdx, -1);
        for (int c = 0; c < 10; c++) {
            int support = 0;
            for (int j = 0; j < 10; j++) support += conf[c][j];
            classAccPct[c] = support == 0 ? 0
                    : 100.0 * conf[c][c] / support;
        }
        for (int i = 0; i < teL.count(); i++)
            if (firstIdx[teL.labels()[i]] == -1)
                firstIdx[teL.labels()[i]] = i;

        puretf.chart.EvalCharts.render(
                "Fashion-MNIST - one sample per class (true label | pred | accuracy)",
                teI.pixels(), teI.rows(), firstIdx,
                teL.labels(), pred, NAMES, classAccPct, showChart,
                "target\\fashion-eval.png");
        System.out.println("evaluation grid saved to target/fashion-eval.png"
                + (showChart ? " and opened in a window" : ""));
    }

    private static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
        return best;
    }

    private static float[] softmax(float[] z) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : z) max = Math.max(max, v);
        double sum = 0;
        float[] p = new float[z.length];
        for (int i = 0; i < z.length; i++) {
            p[i] = (float) Math.exp(z[i] - max);
            sum += p[i];
        }
        for (int i = 0; i < z.length; i++) p[i] /= (float) sum;
        return p;
    }

    private FashionMnistDemo() {}
}
