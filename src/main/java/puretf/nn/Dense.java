package puretf.nn;

import puretf.Tensor;

import java.util.List;
import java.util.Random;

/** Fully-connected layer: out = relu/tanh/none(x . W + b). */
public final class Dense extends Layer {

    public enum Act { NONE, RELU, TANH, SIGMOID }

    private final int units;
    private final Act act;
    private final long seed;
    private Tensor w, b;

    public Dense(int units) { this(units, Act.NONE); }

    public Dense(int units, Act act) {
        this.units = units;
        this.act = act;
        this.seed = seedCounter++;
    }

    private static long seedCounter = 0;

    @Override
    protected void init(int fanIn) {
        Random rnd = new Random(seed);
        w = Tensor.glorot(rnd, fanIn, units, fanIn, units);
        b = Tensor.param(new float[units], units);
    }

    @Override
    protected Tensor apply(Tensor x) {
        Tensor z = x.matmul(w).addBias(b);
        return switch (act) {
            case RELU -> z.relu();
            case TANH -> z.tanh();
            case SIGMOID -> z.sigmoid();
            case NONE -> z;
        };
    }

    @Override
    public List<Tensor> params() { return List.of(w, b); }

    public int units() { return units; }

    @Override public int typeCode() { return 1; }

    @Override public int[] specArgs() { return new int[]{units, act.ordinal()}; }
}
