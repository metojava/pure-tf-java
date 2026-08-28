package puretf.nn;

/** Per-epoch training curves, Keras-style. */
public record History(double[] loss, double[] acc,
                      double[] valLoss, double[] valAcc) {

    public int epochs() { return loss.length; }

    public void printSummary() {
        int n = loss.length;
        System.out.printf("first epoch: loss %.4f%s | last: loss %.4f%s%n",
                loss[0], acc == null ? "" : String.format(" - acc %.4f", acc[0]),
                loss[n - 1],
                acc == null ? "" : String.format(" - acc %.4f", acc[n - 1]));
        if (valLoss != null)
            System.out.printf("last validation: loss %.4f%s%n", valLoss[n - 1],
                    valAcc == null ? "" : String.format(" - acc %.4f", valAcc[n - 1]));
    }
}
