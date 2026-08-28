package puretf.nn;

/** Single-head self-attention with residual: [B,T,D] -> [B,T,D]. */
public final class SelfAttention extends MultiHeadAttention {

    public SelfAttention() { this(false); }

    public SelfAttention(boolean causal) { super(1, causal); }

    @Override public int typeCode() { return 20; }

    @Override public int[] specArgs() { return new int[]{causal() ? 1 : 0}; }
}
