package puretf.demo;

import puretf.Optim;
import puretf.chart.EvalCharts;
import puretf.chart.TrainingCurves;
import puretf.data.IdxDataset;
import puretf.nn.*;

/**
 * Fashion-MNIST classifier - everything computed by the pure-Java autograd
 * engine.
 *
 * <p>The Keras "quickstart" example model, translated to this engine:
 * <pre>
 *   model = tf.keras.Sequential([
 *       tf.keras.layers.Flatten(input_shape=(28, 28)),
 *       tf.keras.layers.Dense(128, activation='relu'),
 *       tf.keras.layers.Dense(10)
 *   ])
 * </pre>
 * Here {@code Flatten(input_shape=(28,28))} is expressed as
 * {@code Reshape(-1, 28, 28)} followed by {@code Flatten()}.
 *
 * Usage: run.cmd fashion-cnn [epochs=5] [--fast]
 *   --fast trains on a 12k/2k subset so a first result lands in minutes.
 */
public final class CnnFashionMnistDemo {

    private static final String[] NAMES = {
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    public static void main(String[] args) throws Exception {
        String dir = "C:\\Users\\dell\\Desktop\\data\\fashion";
        int epochs = 5;
        boolean fast = false;
        float lr = 0.0015f;
        String schedule = "cosine";
        boolean showChart = true;
        for (String a : args) {
            if (a.matches("\\d+")) epochs = Integer.parseInt(a);
            else if (a.equals("--fast")) fast = true;
            else if (a.equals("--nochart")) showChart = false;
            else if (a.startsWith("--lr=")) lr = Float.parseFloat(a.substring(5));
            else if (a.startsWith("--schedule=")) schedule = a.substring(11);
            else dir = a;
        }
        final float LR = lr;
        final boolean SHOW_CHART = showChart;

        long t0 = System.currentTimeMillis();
        var trI = IdxDataset.loadImages(dir + "\\train-images-idx3-ubyte.gz");
        var trL = IdxDataset.loadLabels(dir + "\\train-labels-idx1-ubyte.gz");
        var teI = IdxDataset.loadImages(dir + "\\t10k-images-idx3-ubyte.gz");
        var teL = IdxDataset.loadLabels(dir + "\\t10k-labels-idx1-ubyte.gz");

        float[][] trainX = trI.pixels();
        int[] trainY = trL.labels();
        float[][] testX = teI.pixels();
        int[] testY = teL.labels();
        if (fast) {
            trainX = java.util.Arrays.copyOf(trainX, 12000);
            trainY = java.util.Arrays.copyOf(trainY, 12000);
            testX = java.util.Arrays.copyOf(testX, 2000);
            testY = java.util.Arrays.copyOf(testY, 2000);
        }
        System.out.printf("loaded %d train / %d test 28x28 images (%.1fs)%n",
                trainX.length, testX.length,
                (System.currentTimeMillis() - t0) / 1000.0);

        // ------------------------------------------------------------------
        // Original CNN architecture (kept as a reference).
        // ------------------------------------------------------------------
        // Sequential net = new Sequential()
        //         .add(new Reshape(-1, 28, 28, 1))
        //         .add(new Conv2D(8, 3, 3, 1, 1, Padding.SAME))
        //         .add(new ReLU())
        //         .add(new MaxPool2D(2, 2, 2, 2, Padding.VALID))
        //         .add(new Conv2D(16, 3, 3, 1, 1, Padding.SAME))
        //         .add(new ReLU())
        //         .add(new MaxPool2D(2, 2, 2, 2, Padding.VALID))
        //         .add(new Flatten())
        //         .add(new Dense(128, Dense.Act.RELU))
        //         .add(new Dropout(0.15f))
        //         .add(new Dense(10));

        // Simple, fast MLP with a second hidden layer + dropout for
        // regularisation: Flatten -> Dense(128, relu) -> Dropout(0.2)
        // -> Dense(64, relu) -> Dense(10).
        Sequential net = new Sequential()
                .add(new Reshape(-1, 28, 28))
                .add(new Flatten())
                .add(new Dense(128, Dense.Act.RELU))
                .add(new Dropout(0.2f))
                .add(new Dense(64, Dense.Act.RELU))
                .add(new Dense(10));

        LrSchedule lrs;
        switch (schedule) {
            case "constant": lrs = LrSchedule.constant(LR); break;
            case "step":     lrs = LrSchedule.step(LR, 3); break;
            case "exponential": lrs = LrSchedule.exponential(LR, 0.7f); break;
            default:         lrs = LrSchedule.cosine(LR, LR * (0.0002f / 0.0015f), epochs);
        }

        long trainStart = System.currentTimeMillis();
        History hist = net.fitHistory(trainX, trainY, 10, epochs, 128,
                new Optim.Adam(LR), lrs, 0f, null, true);
        double trainSec = (System.currentTimeMillis() - trainStart) / 1000.0;

        double[] eval = net.evaluate(testX, testY, 10);
        int[][] conf = new int[10][10];
        float[][] logits = net.predict(testX);
        int[] pred = new int[testY.length];
        for (int i = 0; i < testX.length; i++) {
            pred[i] = Metrics.argmax(logits[i]);
            conf[testY[i]][pred[i]]++;
        }
        System.out.println();

        // Compact, machine-readable result line for experiment aggregation.
        System.out.println("RESULT " + String.join("|",
                schedule, Float.toString(LR), Integer.toString(epochs),
                fast ? "12k" : "60k",
                String.format("%.1f", trainSec / epochs),   // s/epoch
                String.format("%.4f", hist.loss()[hist.loss().length - 1]),
                String.format("%.4f", eval[1])));           // test accuracy

        // Prints the report (scores, confusion matrix, per-class metrics) and
        // opens/swrites a pure-Java2D grid (one sample per class, with true
        // label, predicted label and accuracy % under each image).
        Metrics.printReportWithChart(eval[0], eval[1], conf, NAMES,
                testX, 28, testY, pred,
                "Fashion-MNIST - one sample per class (true label | pred | accuracy)",
                SHOW_CHART, "target\\fashion-eval.png");

        net.saveTo(java.nio.file.Path.of("target\\fashion-cnn.bin"));
        TrainingCurves.save("Fashion-MNIST CNN - training curves",
                hist, "target\\fashion-cnn-curves.png");
        System.out.println("\nmodel saved to target/fashion-cnn.bin"
                + "\ncurves saved to target/fashion-cnn-curves.png"
                + "\nevaluation grid saved to target/fashion-eval.png (window opened)");
    }

    private CnnFashionMnistDemo() {}
}
