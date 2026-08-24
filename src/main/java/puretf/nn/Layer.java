package puretf.nn;

import puretf.Tensor;

import java.util.List;

/**
 * Base class of all layers. Weights are initialized LAZILY on first forward,
 * so the input dimension never has to be declared up front - the first batch
 * teaches the layer its fan-in.
 */
public abstract class Layer {

    /** Trainable parameter tensors owned by this layer. */
    public abstract List<Tensor> params();

    /** One-time weight initialization once the input width is known. */
    protected abstract void init(int fanIn);

    /** Apply the transform (called after {@link #init}). */
    protected abstract Tensor apply(Tensor x);

    private int fanIn = -1;

    /** Forward: initializes on first use, then transforms. */
    public Tensor forward(Tensor x) {
        if (fanIn == -1) {
            fanIn = x.shape[x.shape.length - 1];
            init(fanIn);
        }
        return apply(x);
    }

    /** Optional hook for layers with train/inference modes (e.g. Dropout). */
    public void setTraining(boolean training) {}

    protected int fanIn() { return fanIn; }

    // ---------------------- persistence hooks ----------------------------

    /** Serialization type code; -1 marks a layer that cannot be persisted. */
    public int typeCode() { return -1; }

    /** Constructor arguments needed to rebuild this layer on load. */
    public int[] specArgs() { return new int[0]; }

    /** Extra non-tensor buffers (e.g. BatchNorm running stats). */
    public void saveState(java.io.DataOutputStream out) throws java.io.IOException {}

    /** Mirror of {@link #saveState}. */
    public void loadState(java.io.DataInputStream in) throws java.io.IOException {}
}
