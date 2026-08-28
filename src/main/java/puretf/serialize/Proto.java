package puretf.serialize;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Minimal Protocol Buffers wire-format reader/writer (proto2/wire type) used
 * to parse and emit ONNX model files without any external dependency.
 *
 * <p>Supports the field types ONNX needs: varint, 64-bit, length-delimited and
 * 32-bit, with (un)signed varint handling. Only sufficient for TensorProto /
 * GraphProto / ModelProto — not a general protobuf library.</p>
 */
final class Proto {

    private Proto() {}

    // ------------------------------- reader --------------------------------

    /** One decoded field. */
    static final class Field {
        final int number;
        final int wire;     // 0 varint, 1 64-bit, 2 len, 5 32-bit
        final long varint;
        final byte[] bytes;
        final long fixed64;
        final int fixed32;
        Field(int number, int wire, long varint, byte[] bytes, long f64, int f32) {
            this.number = number; this.wire = wire; this.varint = varint;
            this.bytes = bytes; this.fixed64 = f64; this.fixed32 = f32;
        }
        public long varint() { return varint; }
        public byte[] bytes() { return bytes; }
        public String str() { return new String(bytes, StandardCharsets.UTF_8); }
    }

    /** Parse a buffer into a flat list of top-level fields. */
    static List<Field> parseFields(byte[] buf) {
        return parseFields(buf, 0, buf.length);
    }

    static List<Field> parseFields(byte[] buf, int off, int len) {
        List<Field> out = new ArrayList<>();
        IntRef p = new IntRef(off);
        int end = off + len;
        while (p.v < end) {
            long key = readVarintAt(buf, p, end);
            int number = (int)(key >>> 3);
            int wire = (int)(key & 7);
            switch (wire) {
                case 0: out.add(new Field(number, wire, readVarintAt(buf, p, end), null, 0, 0)); break;
                case 1: out.add(new Field(number, wire, 0, null, readFixed64(buf, p), 0)); break;
                case 2: {
                    long n = readVarintAt(buf, p, end);
                    byte[] b = new byte[(int) n];
                    System.arraycopy(buf, p.v, b, 0, (int) n);
                    p.v += n;
                    out.add(new Field(number, wire, 0, b, 0, 0));
                    break;
                }
                case 5: out.add(new Field(number, wire, 0, null, 0, readFixed32(buf, p))); break;
                default: throw new IllegalArgumentException("unsupported proto wire type " + wire);
            }
        }
        return out;
    }

    static final class IntRef { int v; IntRef(int v){this.v=v;} }

    private static long readVarintAt(byte[] b, IntRef p, int end) {
        long v = 0;
        int shift = 0;
        while (true) {
            if (p.v >= end) throw new IllegalArgumentException("varint overflow");
            byte x = b[p.v++];
            v |= (long)(x & 0x7F) << shift;
            if ((x & 0x80) == 0) break;
            shift += 7;
            if (shift > 63) throw new IllegalArgumentException("varint too long");
        }
        return v;
    }
    private static long readFixed64(byte[] b, IntRef p) {
        long v = 0;
        for (int i = 0; i < 8; i++) v |= ((long)(b[p.v++] & 0xFF)) << (8 * i);
        return v;
    }
    private static int readFixed32(byte[] b, IntRef p) {
        int v = 0;
        for (int i = 0; i < 4; i++) v |= (b[p.v++] & 0xFF) << (8 * i);
        return v;
    }

    // ------------------------------- writer --------------------------------

    static final class Buffer {
        private final ByteArrayOutputStream bo = new ByteArrayOutputStream();
        void varint(int number, long v) { key(number, 0); uvarint(v); }
        void fixed64(int number, long v) { key(number, 1); for (int i = 0; i < 8; i++) bo.write((int)(v >>> (8 * i)) & 0xFF); }
        void fixed32(int number, int v) { key(number, 5); for (int i = 0; i < 4; i++) bo.write((v >>> (8 * i)) & 0xFF); }
        void bytes(int number, byte[] v) { key(number, 2); uvarint(v.length); bo.write(v, 0, v.length); }
        void string(int number, String s) { bytes(number, s.getBytes(StandardCharsets.UTF_8)); }
        void raw(byte[] v) { bo.write(v, 0, v.length); }
        void message(int number, Buffer m) { bytes(number, m.toByteArray()); }
        byte[] toByteArray() { return bo.toByteArray(); }
        private void key(int number, int wire) { uvarint(((long) number << 3) | wire); }
        private void uvarint(long v) {
            while ((v & ~0x7FL) != 0) { bo.write((int)((v & 0x7F) | 0x80)); v >>>= 7; }
            bo.write((int) v);
        }
    }

    // ----------------------------- convenience -----------------------------

    /** Zig-zag encoding (used for signed int tensors in some flows). */
    static long zigZag(long v) { return (v << 1) ^ (v >> 63); }
    static long deZigZag(long v) { return (v >>> 1) ^ -(v & 1); }
}
