package puretf.serialize;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Reader/writer for the <b>safetensors</b> format
 * (<a href="https://huggingface.co/docs/safetensors">safetensors</a>), the
 * dependency-free, security-safe tensor storage used for sharing models.
 *
 * <p><b>File layout</b></p>
 * <pre>
 *   [8-byte little-endian header length L]
 *   [L-byte JSON header:  {"tensor": {"dtype", "shape", "data_offsets": [a,b)}}
 *     plus optional "__metadata__"]
 *   [raw tensor payloads, concatenated, little-endian, offset-aligned to 8]
 * </pre>
 *
 * <p>Only the tensors (weights) are stored — the format is intentionally
 * weights-only: there is no architecture in a safetensors file. Use it in
 * tandem with the engine's manifest/bundle for a full model.</p>
 *
 * <p>Supported dtypes: F16, F32, F64, I8, I16, I32, I64, U8, BOOL.</p>
 */
public final class SafeTensors {

    private SafeTensors() {}

    /**
     * Write a {@link ModelBundle} to a binary safetensors file. Order and names
     * of the tensors are preserved from the bundle.
     */
    public static void write(Path path, ModelBundle bundle) throws IOException {
        // Build the JSON header with correct absolute data_offsets.
        String header = headerJson(bundle);
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);

        byte[] payload = new byte[(int) totalPayload(bundle)];
        int off = 0;
        for (NamedTensor t : bundle.tensors()) {
            t.writeLE(payload, off);
            off += t.byteLength();
        }

        byte[] len = new byte[8];
        putLongLE(len, 0, headerBytes.length);
        try (var out = new java.io.BufferedOutputStream(Files.newOutputStream(path))) {
            out.write(len);
            out.write(headerBytes);
            out.write(payload);
        }
    }

    /** Read a safetensors file into a {@link ModelBundle}. */
    public static ModelBundle read(Path path) throws IOException {
        byte[] data = Files.readAllBytes(path);
        if (data.length < 8)
            throw new IOException("safetensors: file too small");
        long hlen = getLongLE(data, 0);
        if (hlen < 0 || hlen + 8 > data.length)
            throw new IOException("safetensors: bad header length " + hlen);
        String json = new String(data, 8, (int) hlen, StandardCharsets.UTF_8);
        int pos = 8 + (int) hlen;

        JsonObj root = JsonObj.parse(json);
        ModelBundle b = new ModelBundle();
        for (String key : root.order) {
            Object v = root.fields.get(key);
            if (key.equals("__metadata__")) continue;
            if (!(v instanceof JsonObj)) continue;
            JsonObj t = (JsonObj) v;
            String dtype = t.str("dtype");
            long[] shapeL = t.longs("shape");
            long[] offs = t.longs("data_offsets");
            int[] shape = new int[shapeL.length];
            for (int i = 0; i < shape.length; i++) shape[i] = (int) shapeL[i];
            long start = 8 + hlen + offs[0];
            long end = 8 + hlen + offs[1];
            if (end > data.length) throw new IOException("safetensors: tensor " + key + " out of range");
            NamedTensor nt = decode(key, dtype, shape, data, (int) start, (int) end);
            b.add(nt);
        }
        return b;
    }

    // ------------------------------------------------------------------ //

    private static String headerJson(ModelBundle bundle) {
        StringBuilder sb = new StringBuilder();
        sb.append('{');
        boolean first = true;
        long base = 0;
        for (NamedTensor t : bundle.tensors()) {
            if (!first) sb.append(',');
            first = false;
            sb.append(quote(t.name())).append(":{");
            sb.append("\"dtype\":").append(quote(dtypeName(t.dtype()))).append(',');
            sb.append("\"shape\":[");
            int[] s = t.shape();
            for (int i = 0; i < s.length; i++) { if (i > 0) sb.append(','); sb.append(s[i]); }
            long len = t.byteLength();
            sb.append("],\"data_offsets\":[").append(base).append(',').append(base + len).append(']');
            sb.append('}');
            base += pad8(len);
        }
        sb.append('}');
        return sb.toString();
    }

    private static long totalPayload(ModelBundle bundle) {
        long total = 0;
        for (NamedTensor t : bundle.tensors()) total += pad8(t.byteLength());
        return total;
    }

    private static long pad8(long n) {
        return (n + 7) & ~7L;
    }

    private static NamedTensor decode(String name, String dtype, int[] shape,
                                      byte[] data, int start, int end)
            throws IOException {
        int count = 1;
        for (int s : shape) count *= s;
        int bytes = end - start;
        if (bytes == 0 && count == 0) { count = 1; } // scalars: shape []
        long[] raw = new long[Math.max(count, 1)];
        switch (dtype) {
            case "F32": {
                if (bytes < count * 4) throw new IOException("safetensors: F32 payload too small for " + name);
                for (int i = 0; i < count; i++)
                    raw[i] = Float.floatToRawIntBits(getFloatLE(data, start + i * 4));
                return NamedTensor.fromRaw(name, shape, NamedTensor.DType.FLOAT32, raw);
            }
            case "F16": {
                if (bytes < count * 2) throw new IOException("safetensors: F16 payload too small for " + name);
                for (int i = 0; i < count; i++)
                    raw[i] = (short) (getIntLE(data, start + i * 2) & 0xFFFF); // half bits
                return NamedTensor.fromRaw(name, shape, NamedTensor.DType.FLOAT16, raw);
            }
            case "F64": {
                if (bytes < count * 8) throw new IOException("safetensors: F64 payload too small for " + name);
                for (int i = 0; i < count; i++)
                    raw[i] = Double.doubleToRawLongBits(getDoubleLE(data, start + i * 8));
                return NamedTensor.fromRaw(name, shape, NamedTensor.DType.FLOAT64, raw);
            }
            case "I64": {
                for (int i = 0; i < count; i++) raw[i] = getLongLE(data, start + i * 8);
                return NamedTensor.fromRaw(name, shape, NamedTensor.DType.INT64, raw);
            }
            case "I32": {
                for (int i = 0; i < count; i++) raw[i] = getIntLE(data, start + i * 4);
                return NamedTensor.fromRaw(name, shape, NamedTensor.DType.INT32, raw);
            }
            case "I16": {
                for (int i = 0; i < count; i++) raw[i] = (short) getIntLE(data, start + i * 2);
                return NamedTensor.fromRaw(name, shape, NamedTensor.DType.INT16, raw);
            }
            case "I8": {
                for (int i = 0; i < count; i++) raw[i] = (byte) data[start + i];
                return NamedTensor.fromRaw(name, shape, NamedTensor.DType.INT8, raw);
            }
            case "U8": {
                for (int i = 0; i < count; i++) raw[i] = data[start + i] & 0xFF;
                return NamedTensor.fromRaw(name, shape, NamedTensor.DType.UINT8, raw);
            }
            case "BOOL": {
                for (int i = 0; i < count; i++) raw[i] = data[start + i] != 0 ? 1 : 0;
                return NamedTensor.fromRaw(name, shape, NamedTensor.DType.BOOL, raw);
            }
            default:
                throw new IOException("safetensors: unsupported dtype " + dtype + " for " + name);
        }
    }

    private static int getIntLE(byte[] b, int o) {
        return (b[o] & 0xFF) | ((b[o + 1] & 0xFF) << 8) |
                ((b[o + 2] & 0xFF) << 16) | ((b[o + 3] & 0xFF) << 24);
    }

    private static double getDoubleLE(byte[] b, int o) {
        long v = 0;
        for (int i = 7; i >= 0; i--) v = (v << 8) | (b[o + i] & 0xFF);
        return Double.longBitsToDouble(v);
    }

    private static float getFloatLE(byte[] b, int o) {
        return Float.intBitsToFloat(
                (b[o] & 0xFF) | ((b[o + 1] & 0xFF) << 8) |
                        ((b[o + 2] & 0xFF) << 16) | ((b[o + 3] & 0xFF) << 24));
    }

    private static void putLongLE(byte[] b, int o, long v) {
        b[o] = (byte) v; b[o + 1] = (byte) (v >> 8); b[o + 2] = (byte) (v >> 16);
        b[o + 3] = (byte) (v >> 24); b[o + 4] = (byte) (v >> 32);
        b[o + 5] = (byte) (v >> 40); b[o + 6] = (byte) (v >> 48); b[o + 7] = (byte) (v >> 56);
    }

    private static long getLongLE(byte[] b, int o) {
        long v = 0;
        for (int i = 7; i >= 0; i--) v = (v << 8) | (b[o + i] & 0xFF);
        return v;
    }

    private static String quote(String s) { return "\"" + s + "\""; }

    private static String dtypeName(NamedTensor.DType d) {
        switch (d) {
            case FLOAT16: return "F16";
            case FLOAT32: return "F32";
            case FLOAT64: return "F64";
            case INT8: return "I8";
            case INT16: return "I16";
            case INT32: return "I32";
            case INT64: return "I64";
            case UINT8: return "U8";
            case BOOL: return "BOOL";
            default: return "F32";
        }
    }

    // -------------------------------------------------------------------- //
    // Minimal JSON parser (header only). No external dependency.

    static final class JsonObj {
        java.util.LinkedHashMap<String, Object> fields = new java.util.LinkedHashMap<>();
        java.util.ArrayList<String> order = new java.util.ArrayList<>();

        String str(String k) { Object v = fields.get(k); return v == null ? null : (String) v; }
        long[] longs(String k) { return (long[]) fields.get(k); }

        static JsonObj parse(String s) {
            int[] idx = { 0 };
            JsonObj root = new JsonObj();
            parseObject(s, idx, root);
            return root;
        }

        private static void parseObject(String s, int[] idx, JsonObj out) {
            skipWs(s, idx);
            expect(s, idx, '{');
            skipWs(s, idx);
            if (peek(s, idx) == '}') { idx[0]++; return; }
            while (true) {
                skipWs(s, idx);
                String key = parseString(s, idx);
                skipWs(s, idx);
                expect(s, idx, ':');
                skipWs(s, idx);
                Object val = parseValue(s, idx);
                out.fields.put(key, val);
                out.order.add(key);
                skipWs(s, idx);
                char c = peek(s, idx);
                if (c == ',') { idx[0]++; continue; }
                if (c == '}') { idx[0]++; break; }
                throw new IllegalArgumentException("bad json near " + idx[0]);
            }
        }

        private static Object parseValue(String s, int[] idx) {
            skipWs(s, idx);
            char c = peek(s, idx);
            if (c == '{') { JsonObj o = new JsonObj(); parseObject(s, idx, o); return o; }
            if (c == '[') {
                idx[0]++;
                java.util.ArrayList<Long> list = new java.util.ArrayList<>();
                skipWs(s, idx);
                if (peek(s, idx) == ']') { idx[0]++; return toArray(list); }
                while (true) {
                    skipWs(s, idx);
                    list.add(Long.parseLong(parseNumber(s, idx)));
                    skipWs(s, idx);
                    char cc = peek(s, idx);
                    if (cc == ',') { idx[0]++; continue; }
                    if (cc == ']') { idx[0]++; break; }
                    throw new IllegalArgumentException("bad json array");
                }
                return toArray(list);
            }
            if (c == '"') return parseString(s, idx);
            return Long.parseLong(parseNumber(s, idx));
        }

        private static long[] toArray(java.util.ArrayList<Long> a) {
            long[] r = new long[a.size()];
            for (int i = 0; i < a.size(); i++) r[i] = a.get(i);
            return r;
        }

        private static String parseString(String s, int[] idx) {
            expect(s, idx, '"');
            StringBuilder sb = new StringBuilder();
            while (idx[0] < s.length()) {
                char c = s.charAt(idx[0]++);
                if (c == '"') return sb.toString();
                if (c == '\\') {
                    char e = s.charAt(idx[0]++);
                    switch (e) {
                        case 'n': sb.append('\n'); break;
                        case 't': sb.append('\t'); break;
                        case 'r': sb.append('\r'); break;
                        case '/': sb.append('/'); break;
                        case '\\': sb.append('\\'); break;
                        case '"': sb.append('"'); break;
                        case 'u': sb.append((char) Integer.parseInt(s.substring(idx[0], idx[0] + 4), 16)); idx[0] += 4; break;
                        default: sb.append(e);
                    }
                } else sb.append(c);
            }
            throw new IllegalArgumentException("unterminated string");
        }

        private static String parseNumber(String s, int[] idx) {
            int st = idx[0];
            while (idx[0] < s.length() && (Character.isDigit(s.charAt(idx[0]))
                    || "-+.eE".indexOf(s.charAt(idx[0])) >= 0)) idx[0]++;
            return s.substring(st, idx[0]);
        }

        private static void skipWs(String s, int[] idx) {
            while (idx[0] < s.length() && Character.isWhitespace(s.charAt(idx[0]))) idx[0]++;
        }
        private static char peek(String s, int[] idx) {
            if (idx[0] >= s.length()) throw new IllegalArgumentException("json ended early");
            return s.charAt(idx[0]);
        }
        private static void expect(String s, int[] idx, char c) {
            if (peek(s, idx) != c) throw new IllegalArgumentException("expected " + c + " at " + idx[0]);
            idx[0]++;
        }
    }
}
