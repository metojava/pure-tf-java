package puretf.serialize;

import java.util.Arrays;

/**
 * A named multi-dimensional tensor used as the intermediate representation
 * for model interchange. Carries its original dtype so that numeric data can
 * be preserved through a write/read cycle where the target format supports it
 * (e.g. safetensors F16, ONNX float16/int64).
 *
 * <p>Data is exposed as a flat array of doubles for maximum precision during
 * the exchange (the engine itself works in floats; external formats often use
 * fp16/int64). Use the {@link #fromFloats} constructor when moving engine
 * parameters (the engine's float[] is reused by reference where possible).
 */
public final class NamedTensor {

    public enum DType {
        FLOAT16(10, 2), FLOAT32(1, 4), FLOAT64(11, 8),
        INT8(3, 1), INT16(5, 2), INT32(6, 4), INT64(7, 8),
        UINT8(2, 1), BOOL(9, 1);

        /** Stable numeric code (ONNX TensorProto data_type where sensible). */
        public final int code;
        /** Bytes per element. */
        public final int bytes;

        DType(int code, int bytes) { this.code = code; this.bytes = bytes; }

        /** True for integral storage (int8..int64, bool). */
        public boolean isInt() {
            return this == INT8 || this == INT16 || this == INT32 ||
                    this == INT64 || this == UINT8;
        }
        public boolean isFloat() { return !isInt() && this != BOOL; }

        public static DType fromCode(int code) {
            for (DType d : values()) if (d.code == code) return d;
            throw new IllegalArgumentException("unknown tensor data code " + code);
        }
    }

    private final String name;
    private final int[] shape;
    private final DType dtype;
    private final long[] data;   // raw storage; floats bit/scale as needed

    private NamedTensor(String name, int[] shape, DType dtype, long[] data) {
        this.name = name;
        this.shape = shape.clone();
        this.dtype = dtype;
        this.data = data;
    }

    /** Wrap an engine float[] by reference. Shape must be non-empty. */
    public static NamedTensor fromFloats(String name, int[] shape, float[] floats) {
        long[] raw = new long[floats.length];
        for (int i = 0; i < floats.length; i++) raw[i] = Float.floatToRawIntBits(floats[i]);
        return new NamedTensor(name, shape, DType.FLOAT32, raw);
    }

    /** Create from float values (converted to float32 storage). */
    public static NamedTensor ofFloat(String name, int[] shape, double[] vals) {
        long[] raw = new long[vals.length];
        for (int i = 0; i < vals.length; i++) raw[i] = Float.floatToRawIntBits((float) vals[i]);
        return new NamedTensor(name, shape, DType.FLOAT32, raw);
    }

    /** Create from decoded raw slots with an explicit dtype (used by readers). */
    public static NamedTensor fromRaw(String name, int[] shape, DType dtype, long[] raw) {
        return new NamedTensor(name, shape, dtype, raw);
    }

    /** Create an integer tensor (int64 storage). */
    public static NamedTensor ofLong(String name, int[] shape, long[] vals, boolean forceInt64) {
        DType d = forceInt64 || needsInt64(vals) ? DType.INT64 : DType.INT32;
        long[] raw = new long[vals.length];
        System.arraycopy(vals, 0, raw, 0, vals.length);
        return new NamedTensor(name, shape, d, raw);
    }

    private static boolean needsInt64(long[] v) {
        for (long x : v) if (x != (int) x) return true;
        return false;
    }

    // ---- accessors ----

    public String name() { return name; }
    public int[] shape() { return shape.clone(); }
    /** The shape array itself (do not mutate). */
    public int[] shapeRef() { return shape; }
    public DType dtype() { return dtype; }

    public int rank() { return shape.length; }
    public int numel() { return data.length; }

    public long elementCount() {
        long n = 1;
        for (int s : shape) n *= s;
        return n;
    }

    /** Raw 64-bit slots; interpret per {@link #dtype}. */
    public long[] raw() { return data; }

    /** Read element {@code i} as a double (floats/exact ints; bool 0/1). */
    public double doubleAt(int i) {
        switch (dtype) {
            case FLOAT16: return fromHalf((short) data[i]);
            case FLOAT32: return Float.intBitsToFloat((int) data[i]);
            case FLOAT64: return Double.longBitsToDouble(data[i]);
            case INT8:  return (byte) data[i];
            case INT16: return (short) data[i];
            case INT32: return (int) data[i];
            case INT64: return data[i];
            case UINT8: return data[i] & 0xFF;
            case BOOL:  return data[i] != 0 ? 1 : 0;
            default: throw new IllegalStateException();
        }
    }

    /** Read element {@code i} as a long. */
    public long longAt(int i) { return (long) doubleAt(i); }

    /** Convenience: full double array. */
    public double[] toDoubles() {
        double[] out = new double[data.length];
        for (int i = 0; i < out.length; i++) out[i] = doubleAt(i);
        return out;
    }

    /** Convenience: full float array (lossy for float64/int64 beyond float. */
    public float[] toFloats() {
        float[] out = new float[data.length];
        for (int i = 0; i < out.length; i++) out[i] = (float) doubleAt(i);
        return out;
    }

    /** Convenience: full int array. */
    public int[] toInts() {
        int[] out = new int[data.length];
        for (int i = 0; i < out.length; i++) out[i] = (int) doubleAt(i);
        return out;
    }

    public NamedTensor renamed(String newName) {
        return new NamedTensor(newName, shape, dtype, data);
    }

    // ---- little-endian byte helpers (common to all encoders) ----

    /** Total payload bytes for this tensor given its dtype. */
    public long byteLength() { return (long) data.length * dtype.bytes; }

    /** Serialize into {@code buf} starting at {@code off}, little-endian. */
    public void writeLE(byte[] buf, int off) {
        switch (dtype) {
            case FLOAT16:
                for (long v : data) { short h = toHalf((float) doubleAt_(v)); buf[off++]=(byte)h; buf[off++]=(byte)(h>>8); }
                break;
            case FLOAT32:
                for (long v : data) { int f = Float.floatToRawIntBits(Float.intBitsToFloat((int) v)); buf[off++]= (byte)f; buf[off++]=(byte)(f>>8); buf[off++]=(byte)(f>>16); buf[off++]=(byte)(f>>24); }
                break;
            case FLOAT64:
                for (long v : data) { putLongLE(buf, off, v); off += 8; }
                break;
            case INT8:  case BOOL:
                for (long v : data) buf[off++] = (byte) v;
                break;
            case UINT8:
                for (long v : data) buf[off++] = (byte) v;
                break;
            case INT16:
                for (long v : data) { buf[off++]=(byte)v; buf[off++]=(byte)(v>>8); }
                break;
            case INT32:
                for (long v : data) { putIntLE(buf, off, (int) v); off += 4; }
                break;
            case INT64:
                for (long v : data) { putLongLE(buf, off, v); off += 8; }
                break;
            default: throw new IllegalStateException();
        }
    }

    private static void putIntLE(byte[] b, int o, int v) {
        b[o]= (byte)v; b[o+1]=(byte)(v>>8); b[o+2]=(byte)(v>>16); b[o+3]=(byte)(v>>24);
    }
    private static void putLongLE(byte[] b, int o, long v) {
        b[o]=  (byte)v; b[o+1]=(byte)(v>>8); b[o+2]=(byte)(v>>16); b[o+3]=(byte)(v>>24);
        b[o+4]=(byte)(v>>32); b[o+5]=(byte)(v>>40); b[o+6]=(byte)(v>>48); b[o+7]=(byte)(v>>56);
    }

    private double doubleAt_(long v) {
        // force re-read through dtype (int32/64 stored raw, floats need bits)
        switch (dtype) {
            case FLOAT16: return fromHalf((short) v);
            case FLOAT32: return Float.intBitsToFloat((int) v);
            default: return v;
        }
    }

    // ---- half precision (IEEE 754 binary16) ----

    public static short toHalf(float val) {
        int x = Float.floatToRawIntBits(val);
        int sign = (x >>> 16) & 0x8000;
        int abs = x & 0x7fffffff;
        if (abs >= 0x7f800000) {              // inf/NaN
            return (short)(sign | 0x7c00 | ((abs & 0x3ffff) != 0 ? 0x200 : 0));
        }
        int mant = abs & 0x7fffff;
        int exp  = (abs >>> 23) & 0xff;
        int half;
        if (exp == 0) {                       // zero / subnormal
            half = 0;
        } else if (exp - 127 > 15) {          // overflow -> inf
            half = 0x7c00;
        } else if (exp - 127 < -14) {         // underflow
            half = 0;
        } else {
            half = ((exp - 127) + 15) << 10;
            half |= mant >>> 13;
            if ((mant & 0x1fff) != 0) {       // round-to-nearest-even-ish
                int rem = mant & 0x1fff;
                if (rem > 0x1000 || (rem == 0x1000 && (half & 1) != 0)) half++;
            }
        }
        return (short)(sign | half);
    }

    public static float fromHalf(short h) {
        int sign = (h & 0x8000) << 16;
        int exp  = (h >>> 10) & 0x1f;
        int mant = h & 0x3ff;
        int f;
        if (exp == 0) {
            f = sign | (mant << 13);          // subnormal -> normal float
        } else if (exp == 31) {               // inf/NaN
            f = sign | 0x7f800000 | (mant << 13);
        } else {
            f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        return Float.intBitsToFloat(f);
    }

    @Override public String toString() {
        return name + " " + Arrays.toString(shape) + " " + dtype + " n=" + data.length;
    }
}
