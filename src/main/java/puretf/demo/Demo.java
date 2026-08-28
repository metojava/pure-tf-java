package puretf.demo;

import puretf.nn.Dense;
import puretf.nn.Dropout;
import puretf.nn.Sequential;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Both demos now use the LAYERS API (Sequential/Dense/Dropout) instead of
 * hand-wiring the tape - the same task set, a fraction of the code.
 */
public final class Demo {

    public static void main(String[] args) {
        xorDemo();
        regressionDemo();
    }

    /* ==================== 1. XOR as classification ===================== */

    private static void xorDemo() {
        System.out.println("=== XOR via Sequential.fitClassifier ===");
        float[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] Y = {0, 1, 1, 0};                       // XOR class ids

        Sequential net = new Sequential()
                .add(new Dense(8, Dense.Act.TANH))
                .add(new Dense(2));                    // logits for 2 classes

        double[] hist = net.fitClassifier(X, Y, 2, 2000, 4, 0.05f);
        System.out.printf("final loss %.6f%n", hist[hist.length - 1]);

        String[] names = {"class0", "class1"};
        int[] pred = net.predictClass(X);
        for (int i = 0; i < X.length; i++)
            System.out.printf("%d^%d = %s%n", (int) X[i][0], (int) X[i][1], names[pred[i]]);
        System.out.println();
    }

    /* ============ 2. Regression with mini-batching + dropout ============ */

    private static void regressionDemo() {
        System.out.println("=== Regression: y = 3*x0 - 2*x1 + 0.5*sin(3*x0) + noise ===");
        Random gen = new Random(42L);
        int n = 600;
        float[][] X = new float[n][2];
        float[][] Y = new float[n][1];
        for (int i = 0; i < n; i++) {
            float a = gen.nextFloat() * 4f - 2f;
            float b = gen.nextFloat() * 4f - 2f;
            X[i][0] = a;
            X[i][1] = b;
            Y[i][0] = (float) (3 * a - 2 * b + 0.5 * Math.sin(3 * a))
                    + (float) gen.nextGaussian() * 0.15f;
        }

        // seeded 80/20 split
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < n; i++) order.add(i);
        java.util.Collections.shuffle(order, new Random(7L));
        int nTrain = n - 120;
        float[][] trainX = new float[nTrain][];
        float[][] trainY = new float[nTrain][];
        float[][] testX = new float[n - nTrain][];
        float[][] testY = new float[n - nTrain][];
        for (int i = 0; i < nTrain; i++) {
            trainX[i] = X[order.get(i)];
            trainY[i] = Y[order.get(i)];
        }
        for (int i = nTrain; i < n; i++) {
            testX[i - nTrain] = X[order.get(i)];
            testY[i - nTrain] = Y[order.get(i)];
        }

        Sequential net = new Sequential()
                .add(new Dense(32, Dense.Act.TANH))
                .add(new Dropout(0.1f))
                .add(new Dense(16, Dense.Act.TANH))
                .add(new Dense(1));

        double[] hist = net.fit(trainX, trainY, 300, 32, 0.01f);
        System.out.printf("train loss: first %.5f -> last %.5f%n",
                hist[0], hist[hist.length - 1]);

        var preds = net.predict(testX);
        double se = 0;
        for (int i = 0; i < testX.length; i++) {
            double e = preds[i][0] - testY[i][0];
            se += e * e;
        }
        System.out.printf("test RMSE %.4f (noise floor ~0.15)%n",
                Math.sqrt(se / testX.length));
    }

    private Demo() {}
}
