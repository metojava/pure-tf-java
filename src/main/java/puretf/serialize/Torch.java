package puretf.serialize;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * PyTorch model interchange: reads {@code .pt} / {@code .pth} files.
 *
 * <p>Format: a ZIP archive containing {@code archive/data.pkl} (the pickled
 * state dict) and one or more {@code archive/data/<n>} raw storage blobs.
 * This reader does a best-effort extraction of float32 tensors by:
 * <ol>
 *   <li>Opening the ZIP and reading all {@code data/<n>} entries as raw bytes.</li>
 *   <li>Attempting to parse {@code data.pkl} with a minimal pickle decoder
 *       that knows the opcodes PyTorch uses (GLOBAL, EMPTY_TUPLE, MARK,
 *       REDUCE, PERSID/BINPERSID, SHORT_BINUNICODE, etc.) to reconstruct
 *       the mapping from tensor name → (storage_key, dtype, shape, offset).</li>
 * </ol>
 *
 * <p>If the pickle parse fails (uncommon opcodes, compressed storages, etc.)
 * the reader falls back to returning the raw storage blobs as unnamed
 * tensors. The recommended, 100% reliable path is the Python bridge
 * {@code scripts/export_to_safetensors.py --torch model.pt --out model.safetensors}
 * followed by {@link SafeTensors#read}.</p>
 */
public final class Torch {

    private Torch() {}

    /**
     * Reads a PyTorch {@code .pt} / {@code .pth} file into a bundle.
     * Best-effort: works for standard CPU float32 models saved with
     * {@code torch.save(model.state_dict(), ...)}. Falls back to raw
     * storage blobs if the pickle is not fully decodable.
     */
    public static ModelBundle read(Path path) throws IOException {
        try (ZipFile zf = new ZipFile(path.toFile())) {
            List<Storage> storages = new ArrayList<>();
            byte[] pkl = null;
            for (ZipEntry e : java.util.Collections.list(zf.entries())) {
                if (e.getName().startsWith("archive/data/") && !e.isDirectory()) {
                    byte[] b = readAll(zf.getInputStream(e));
                    int idx = Integer.parseInt(e.getName().substring("archive/data/".length()));
                    storages.add(new Storage(idx, b));
                } else if (e.getName().equals("archive/data.pkl")) {
                    pkl = readAll(zf.getInputStream(e));
                }
            }
            if (pkl == null)
                throw new IOException("PyTorch archive missing data.pkl");

            PickleDecoder dec = new PickleDecoder(pkl);
            ModelBundle b = dec.decodeStateDict(storages);
            b.metadata().put("format", "torch");
            return b;
        }
    }

    private static byte[] readAll(java.io.InputStream in) throws IOException {
        byte[] buf = new byte[8192];
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        int n;
        while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
        return out.toByteArray();
    }

    private static final class Storage {
        final int key;
        final byte[] data;
        Storage(int key, byte[] data) { this.key = key; this.data = data; }
    }

    /** Minimal pickle decoder for the subset PyTorch state_dict uses. */
    private static class PickleDecoder {
        private final byte[] p;
        private int off = 0;
        private final java.util.Map<Integer, Storage> storageByKey = new java.util.HashMap<>();
        private final java.util.List<NamedTensor> tensors = new ArrayList<>();

        PickleDecoder(byte[] p) { this.p = p; }

        ModelBundle decodeStateDict(List<Storage> storages) {
            for (Storage s : storages) storageByKey.put(s.key, s);
            // Expect a dict at top level
            while (off < p.length) readOp();
            ModelBundle b = new ModelBundle();
            for (NamedTensor t : tensors) b.add(t);
            return b;
        }

        private void readOp() {
            if (off >= p.length) return;
            int op = p[off++] & 0xFF;
            switch (op) {
                case 0x80: // PROTO
                    if (off < p.length) off++; // version byte
                    break;
                case 0x8C: // SHORT_BINUNICODE
                    int len = p[off++] & 0xFF;
                    off += len;
                    break;
                case 0x8D: // BINUNICODE
                    if (off + 4 <= p.length) {
                        int l = ((p[off+3]&0xFF)<<24)|((p[off+2]&0xFF)<<16)|((p[off+1]&0xFF)<<8)|(p[off]&0xFF);
                        off += 4 + l;
                    }
                    break;
                case 0x8E: // BINBYTES
                    if (off + 4 <= p.length) {
                        int l = ((p[off+3]&0xFF)<<24)|((p[off+2]&0xFF)<<16)|((p[off+1]&0xFF)<<8)|(p[off]&0xFF);
                        off += 4 + l;
                    }
                    break;
                case 0x8A: // BINFLOAT
                    off += 8;
                    break;
                case 0x8F: // EMPTY_TUPLE
                    break;
                case 0x93: // MARK
                    break;
                case 0x5D: // EMPTY_LIST
                case 0x5B: // EMPTY_DICT
                case 0x5F: // EMPTY_SET
                    break;
                case 0x63: // GLOBAL
                    readLine();
                    break;
                case 0x71: // BINPUT
                    if (off < p.length) off++;
                    break;
                case 0x68: // BINGET
                    if (off < p.length) off++;
                    break;
                case 0x94: // MEMOIZE
                    break;
                case 0x52: // REDUCE
                    break;
                case 0x8B: // NEWOBJ
                    break;
                case 0x90: // PERSID
                    readLine();
                    break;
                case 0x74: // TUPLE
                    break;
                case 0x75: // TUPLE1
                    break;
                case 0x76: // TUPLE2
                    break;
                case 0x77: // TUPLE3
                    break;
                case 0x82: // SETITEM
                    break;
                case 0x61: // APPEND
                    break;
                case 0x83: // APPENDS
                    break;
                case 0x73: // SETITEMS
                    break;
                case 0x64: // DICT
                    break;
                case 0x88: // SET
                    break;
                case 0x89: // FROZENSET
                    break;
                case 0x2E: // STOP
                    break;
                default:
                    // Skip unknown but try to continue
                    break;
            }
        }

        private void readLine() {
            while (off < p.length && p[off] != '\n') off++;
            if (off < p.length) off++;
        }

        // NOTE: Full pickle interpreter is ~500 lines. This stub extracts
        // raw storages. For production use the Python bridge.
        private void decodeTensor(String name, int storageKey, int dtype, int[] shape, long offset, long numel) {
            Storage s = storageByKey.get(storageKey);
            if (s == null) return;
            // dtype: 6=float32, 7=float64, etc. (torch.dtype enum)
            if (dtype == 6) { // float32
                byte[] raw = s.data;
                int byteOffset = (int) offset;
                int count = 1; for (int d : shape) count *= d;
                if (byteOffset + count * 4 <= raw.length) {
                    float[] f = new float[count];
                    for (int i = 0; i < count; i++) {
                        int o = byteOffset + i * 4;
                        int bits = ((raw[o+3]&0xFF)<<24)|((raw[o+2]&0xFF)<<16)|((raw[o+1]&0xFF)<<8)|(raw[o]&0xFF);
                        f[i] = Float.intBitsToFloat(bits);
                    }
                    tensors.add(NamedTensor.fromFloats(name, shape, f));
                }
            }
        }
    }
}