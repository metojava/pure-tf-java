package puretf.nn;

/** DL4J-style evaluation report computed from a confusion matrix. */
public final class Metrics {

    /** Index of the largest element in the array. */
    public static int argmax(float[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
        return best;
    }

    /**
     * Prints overall scores, confusion matrix and per-class
     * precision/recall/F1/support. {@code conf[true][predicted]}.
     */
    public static void printReport(double loss, double accuracy,
                                   int[][] conf, String[] names) {
        int k = conf.length;
        double[] prec = new double[k], rec = new double[k], f1 = new double[k];
        double precSum = 0, recSum = 0, f1Sum = 0;

        for (int c = 0; c < k; c++) {
            int tp = conf[c][c], fp = 0, fn = 0;
            for (int r = 0; r < k; r++) fp += conf[r][c];
            for (int j = 0; j < k; j++) fn += conf[c][j];
            fp -= tp; fn -= tp;
            prec[c] = tp + fp > 0 ? (double) tp / (tp + fp) : 0;
            rec[c]  = tp + fn > 0 ? (double) tp / (tp + fn) : 0;
            f1[c]   = prec[c] + rec[c] > 0 ? 2 * prec[c] * rec[c] / (prec[c] + rec[c]) : 0;
            precSum += prec[c]; recSum += rec[c]; f1Sum += f1[c];
        }

        System.out.println("==========================Scores========================================");
        System.out.printf(" Loss (sparse categorical cross-entropy): %.4f%n", loss);
        System.out.printf(" Accuracy:        %.4f%n", accuracy);
        System.out.printf(" Precision:       %.4f%n", precSum / k);
        System.out.printf(" Recall:          %.4f%n", recSum / k);
        System.out.printf(" F1 score:        %.4f%n", f1Sum / k);
        System.out.println("========================================================================");

        System.out.println("\nConfusion matrix (rows = true label, columns = predicted):");
        System.out.print("        ");
        for (int c = 0; c < k; c++) System.out.printf("%6d", c);
        System.out.println();
        for (int r = 0; r < k; r++) {
            System.out.printf("%3d     ", r);
            for (int c = 0; c < k; c++) System.out.printf("%6d", conf[r][c]);
            System.out.println();
        }

        System.out.println("\nPer-class metrics:");
        System.out.printf("  %-13s %9s %9s %9s %8s%n",
                "class", "precision", "recall", "f1", "support");
        for (int c = 0; c < k; c++) {
            int support = 0;
            for (int j = 0; j < k; j++) support += conf[c][j];
            System.out.printf("  %-13s %9.4f %9.4f %9.4f %8d%n",
                    names[c], prec[c], rec[c], f1[c], support);
        }
    }

    /** DL4J-style per-cell explanation lines + boxed confusion matrix. */
    public static void printDl4jMatrix(int[][] conf) {
        int k = conf.length;
        System.out.println();
        for (int r = 0; r < k; r++)
            for (int c = 0; c < k; c++)
                if (conf[r][c] > 0)
                    System.out.printf(
                            "Examples labeled as %d classified by model as %d: %d times%n",
                            r, c, conf[r][c]);

        System.out.println("\nConfusion matrix:");
        System.out.print("     ");
        for (int c = 0; c < k; c++) System.out.printf("%5d", c);
        System.out.println();
        StringBuilder line = new StringBuilder("     ");
        for (int c = 0; c < k; c++) line.append("-----");
        line.append('-');
        System.out.println(line);
        for (int r = 0; r < k; r++) {
            System.out.printf("%3d  ", r);
            for (int c = 0; c < k; c++) System.out.printf("%5d", conf[r][c]);
            System.out.println();
        }
    }
}
