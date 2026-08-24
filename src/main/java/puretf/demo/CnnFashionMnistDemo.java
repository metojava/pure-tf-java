package puretf.demo;

import puretf.Optim;
import puretf.chart.EvalCharts;
import puretf.chart.TrainingCurves;
import puretf.data.IdxDataset;
import puretf.nn.*;

/**
 * Fashion-MNIST with a CONVOLUTIONAL network - everything computed by the
 * pure-Java autograd engine (im2col conv, max pooling, Adam).
 *
 * Architecture:
 *   [28x28x1] -> Conv(8,3x3,SAME)+ReLU -> MaxPool(2x2)
 *             -> Conv(16,3x3,SAME)+ReLU -> MaxPool(2x2)
 *             -> Flatten -> Dense(128)+ReLU -> Dropout -> Dense(10)
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
        for (String a : args) {
            if (a.matches("\\d+")) epochs = Integer.parseInt(a);
            else if (a.equals("--fast")) fast = true;
            else dir = a;
        }

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

        Sequential net = new Sequential()
                .add(new Reshape(-1, 28, 28, 1))
                .add(new Conv2D(8, 3, 3, 1, 1, Padding.SAME))
                .add(new ReLU())
                .add(new MaxPool2D(2, 2, 2, 2, Padding.VALID))
                .add(new Conv2D(16, 3, 3, 1, 1, Padding.SAME))
                .add(new ReLU())
                .add(new MaxPool2D(2, 2, 2, 2, Padding.VALID))
                .add(new Flatten())
                .add(new Dense(128, Dense.Act.RELU))
                .add(new Dropout(0.15f))
                .add(new Dense(10));

        History hist = net.fitHistory(trainX, trainY, 10, epochs, 96,
                new Optim.Adam(0.0015f),
                LrSchedule.cosine(0.0015f, 0.0002f, epochs),
                0f, null, true);

        double[] eval = net.evaluate(testX, testY, 10);
        int[][] conf = new int[10][10];
        float[][] logits = net.predict(testX);
        for (int i = 0; i < testX.length; i++) {
            int p = Metrics.argmax(logits[i]);
            conf[testY[i]][p]++;
        }
        System.out.println();
        Metrics.printReport(eval[0], eval[1], conf, NAMES);

        net.saveTo(java.nio.file.Path.of("target\\fashion-cnn.bin"));
        TrainingCurves.save("Fashion-MNIST CNN - training curves",
                hist, "target\\fashion-cnn-curves.png");
        System.out.println("\nmodel saved to target/fashion-cnn.bin"
                + "\ncurves saved to target/fashion-cnn-curves.png");
    }

    private CnnFashionMnistDemo() {}
}
