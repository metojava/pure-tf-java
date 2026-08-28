package puretf.nn;

import puretf.Tensor;

import java.util.List;
import java.util.Random;

/**
 * Token embedding: input [B,T] holding integer token ids (as floats) is
 * looked up in a trainable table and returned as [B,T,D].
 */
public final class Embedding extends Layer {

    private final int vocab, dim;
    private Tensor table;
    private final long seed;

    public Embedding(int vocab, int dim) {
        this.vocab = vocab;
        this.dim = dim;
        this.seed = seedCounter++;
    }

    private static long seedCounter = 0;

    @Override
    protected void init(int fanIn) {
        Random rnd = new Random(seed);
        table = Tensor.glorot(rnd, vocab, dim, vocab, dim);
    }

    @Override
    protected Tensor apply(Tensor x) {
        int b = x.shape[0], t = x.shape[x.shape.length - 1];
        int[] ids = new int[b * t];
        for (int i = 0; i < ids.length; i++) {
            int id = Math.round(x.data[i]);
            if (id < 0 || id >= vocab) throw new IllegalArgumentException(
                    "Embedding: token id " + id + " outside [0," + vocab + ")");
            ids[i] = id;
        }
        return Tensor.embedding(table, ids).reshape(b, t, dim);
    }

    @Override
    public List<Tensor> params() { return List.of(table); }

    @Override public int typeCode() { return 18; }

    @Override public int[] specArgs() { return new int[]{vocab, dim}; }
}
