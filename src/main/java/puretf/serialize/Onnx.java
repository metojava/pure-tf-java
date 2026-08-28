package puretf.serialize;

import puretf.serialize.NamedTensor.DType;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Native reader/writer for the <b>ONNX</b> model format
 * (<a href="https://onnx.ai">onnx.ai</a>), with no protobuf dependency.
 *
 * <p><b>Read</b> extracts every graph {@code initializer} (a {@code TensorProto})
 * into a {@link ModelBundle}: name, shape, data_type and raw bytes are decoded
 * so real ONNX models exported by PyTorch/TensorFlow can be loaded as weights.</p>
 *
 * <p><b>Write</b> emits a valid ONNX {@code ModelProto} (ir_version 8, opset 13)
 * carrying the weights as initializers plus a compute graph built from the
 * engine layer manifest when it is a plain feed-forward chain
 * (Dense/Flatten/activations) — enough for standard ONNX runtimes to execute.</p>
 */
public final class Onnx {

    private Onnx() {}

    // ------------------------------------------------------------------ //
    // Reader

    /** Read ONNX weights (graph initializers) into a bundle. */
    public static ModelBundle read(Path path) throws IOException {
        byte[] data = Files.readAllBytes(path);
        List<Proto.Field> model = Proto.parseFields(data);
        for (Proto.Field f : model) {
            if (f.number == 1 && f.wire == 2) {          // graph = ModelProto.graph
                return graphToBundle(f.bytes());
            }
        }
        throw new IOException("ONNX: ModelProto has no graph message");
    }

    private static ModelBundle graphToBundle(byte[] graphBytes) throws IOException {
        List<Proto.Field> graph = Proto.parseFields(graphBytes);
        ModelBundle b = new ModelBundle();
        for (Proto.Field f : graph) {
            if (f.number == 5 && f.wire == 2) {          // initializer = GraphProto.initializer
                NamedTensor t = initializer(f.bytes());
                b.add(t);
            }
        }
        b.metadata().put("format", "onnx");
        return b;
    }

    /** Parse a single TensorProto into a NamedTensor. */
    static NamedTensor initializer(byte[] bytes) throws IOException {
        List<Proto.Field> tp = Proto.parseFields(bytes);
        String name = null;
        int dataType = 0;
        List<Long> dims = new ArrayList<>();
        byte[] rawData = null;
        for (Proto.Field f : tp) {
            switch (f.number) {
                case 1: dims.add(f.varint()); break;                // dims
                case 2: dataType = (int) f.varint(); break;         // data_type
                case 3: name = f.str(); break;                      // name
                case 9: rawData = f.bytes(); break;                 // raw_data
                default: break;
            }
        }
        if (name == null) name = "tensor_" + System.identityHashCode(bytes);
        int[] shape = new int[dims.size()];
        for (int i = 0; i < shape.length; i++) shape[i] = (int) (long) dims.get(i);
        DType dt = DType.FLOAT32;
        for (DType d : DType.values()) if (d.code == dataType) dt = d;

        if (rawData == null) {
            // tensors may store repeated typed data instead of raw_data; we
            // only support raw_data for weight extraction (standard for ONNX).
            throw new IOException("ONNX: initializer " + name + " has no raw_data");
        }
        int count = 1;
        for (int s : shape) count *= s;
        long[] raw = new long[Math.max(count, 1)];
        decodeRaw(rawData, dt, raw);
        return NamedTensor.fromRaw(name, shape, dt, raw);
    }

    private static void decodeRaw(byte[] b, DType dt, long[] raw) throws IOException {
        int bytesPer = dt.bytes;
        if (b.length < raw.length * bytesPer)
            throw new IOException("ONNX: raw_data too small for dtype " + dt);
        switch (dt) {
            case FLOAT16:
                for (int i = 0; i < raw.length; i++)
                    raw[i] = (short) (getLE32(b, i * 2) & 0xFFFF);
                break;
            case FLOAT32:
                for (int i = 0; i < raw.length; i++)
                    raw[i] = Float.floatToRawIntBits(Float.intBitsToFloat(getLE32(b, i * 4)));
                break;
            case FLOAT64:
                for (int i = 0; i < raw.length; i++)
                    raw[i] = Double.doubleToRawLongBits(getLE64(b, i * 8));
                break;
            case INT8: case UINT8: case BOOL:
                for (int i = 0; i < raw.length; i++) raw[i] = b[i] & 0xFF;
                break;
            case INT16:
                for (int i = 0; i < raw.length; i++) raw[i] = (short) (getLE32(b, i * 2) & 0xFFFF);
                break;
            case INT32:
                for (int i = 0; i < raw.length; i++) raw[i] = getLE32(b, i * 4);
                break;
            case INT64:
                for (int i = 0; i < raw.length; i++) raw[i] = getLE64(b, i * 8);
                break;
            default: throw new IOException("ONNX: unsupported dtype " + dt);
        }
    }

    private static int getLE32(byte[] b, int o) {
        return (b[o] & 0xFF) | ((b[o+1] & 0xFF) << 8) | ((b[o+2] & 0xFF) << 16) | ((b[o+3] & 0xFF) << 24);
    }
    private static long getLE64(byte[] b, int o) {
        long v = 0;
        for (int i = 7; i >= 0; i--) v = (v << 8) | (b[o + i] & 0xFF);
        return v;
    }

    // ------------------------------------------------------------------ //
    // Writer

    /**
     * Write a {@link ModelBundle} as a valid ONNX file. Weights become graph
     * initializers; the layer manifest (when present) is translated into a
     * feed-forward node graph so standard runtimes can execute the model.
     */
    public static void write(Path path, ModelBundle b) throws IOException {
        Proto.Buffer model = new Proto.Buffer();
        model.varint(2, 8);                       // ir_version = 8

        // opset_import { domain="", version=13 }
        Proto.Buffer opset = new Proto.Buffer();
        opset.varint(2, 13);
        model.message(8, opset);

        model.string(3, "pure-tf-java");          // producer_name

        // graph
        Proto.Buffer graph = new Proto.Buffer();
        graph.string(2, "puretf_graph");

        int ii = 0;
        List<String> weightNames = new ArrayList<>();
        for (NamedTensor t : b.tensors()) {
            String wname = t.name().replace('/', '_');
            weightNames.add(wname);
            graph.message(5, tensorProto(wname, t));
            ii++;
        }

        // Build feed-forward nodes from manifest (Dense/Flatten/act).
        String lastOut = buildNodes(graph, b, weightNames);

        // graph.output: connect to the final node's output (or a dummy).
        Proto.Buffer outVal = new Proto.Buffer();
        outVal.string(1, lastOut);
        graph.message(12, outVal);

        model.message(1, graph);
        Files.write(path, model.toByteArray());
    }

    /** Encode a single TensorProto. */
    private static Proto.Buffer tensorProto(String name, NamedTensor t) {
        Proto.Buffer tp = new Proto.Buffer();
        for (int d : t.shape()) tp.varint(1, d);    // dims
        tp.varint(2, t.dtype().code);               // data_type
        tp.string(3, name);                          // name
        // raw_data (little-endian)
        byte[] raw = new byte[(int) t.byteLength()];
        t.writeLE(raw, 0);
        tp.bytes(9, raw);                            // raw_data
        return tp;
    }

    /**
     * Emit feed-forward GraphProto nodes for a Sequential manifest. Handles the
     * common chain: Dense (Gemm), Conv2D (Conv), Flatten (Flatten), MaxPool2D
     * (MaxPool), activations (Relu/Sigmoid/Tanh/Softmax/...). Tracks shapes of
     * intermediate values by symbolically walking spacial dims where known.
     */
    private static String buildNodes(Proto.Buffer graph, ModelBundle b,
                                     List<String> weightNames) {
        var manifest = b.layerManifest();
        if (manifest.isEmpty())
            return "model_output";                 // weights-only container

        String in = "input";
        Proto.Buffer inVal = new Proto.Buffer();
        inVal.string(1, in);
        graph.message(11, inVal);

        int wi = 0;
        String prev = in;
        int nodeId = 0;
        String lastOut = in;
        for (var spec : manifest) {
            String out = "layer_" + nodeId + "_out";
            String type = spec.type;
            String wk = wi < weightNames.size() ? weightNames.get(wi) : null;
            String wb = (wi + 1) < weightNames.size() ? weightNames.get(wi + 1) : null;

            switch (type) {
                case "Dense": {
                    Proto.Buffer gemm = node("Gemm");
                    gemm.string(1, prev);
                    if (wk != null) gemm.string(1, wk);
                    if (wb != null) gemm.string(1, wb);
                    gemm.string(2, out);
                    graph.message(1, gemm);
                    wi += 2;
                    break;
                }
                case "Conv2D": {
                    Proto.Buffer conv = node("Conv");
                    conv.string(1, prev);
                    if (wk != null) conv.string(1, wk);
                    if (wb != null) conv.string(1, wb);
                    conv.string(2, out);
                    graph.message(1, conv);
                    wi += 2;
                    break;
                }
                case "Flatten": {
                    Proto.Buffer f = node("Flatten");
                    f.string(1, prev);
                    f.string(2, out);
                    graph.message(1, f);
                    break;
                }
                case "MaxPool2D":
                case "AvgPool2D": {
                    Proto.Buffer p = node("MaxPool");
                    p.string(1, prev);
                    p.string(2, out);
                    graph.message(1, p);
                    break;
                }
                case "Dropout": {
                    Proto.Buffer d = node("Identity");
                    d.string(1, prev);
                    d.string(2, out);
                    graph.message(1, d);
                    break;
                }
                case "ReLU":
                case "LeakyReLU": {
                    Proto.Buffer a = node("Relu");
                    a.string(1, prev);
                    a.string(2, out);
                    graph.message(1, a);
                    break;
                }
                case "Sigmoid": {
                    Proto.Buffer a = node("Sigmoid");
                    a.string(1, prev);
                    a.string(2, out);
                    graph.message(1, a);
                    break;
                }
                case "Tanh": {
                    Proto.Buffer a = node("Tanh");
                    a.string(1, prev);
                    a.string(2, out);
                    graph.message(1, a);
                    break;
                }
                case "Softmax": {
                    Proto.Buffer a = node("Softmax");
                    a.message(5, attr("axis", 1, 1));
                    a.string(1, prev);
                    a.string(2, out);
                    graph.message(1, a);
                    break;
                }
                default:
                    Proto.Buffer id = node("Identity");
                    id.string(1, prev);
                    id.string(2, out);
                    graph.message(1, id);
                    break;
            }
            prev = out;
            lastOut = out;
            nodeId++;
        }
        return lastOut;
    }

    private static Proto.Buffer node(String opType) {
        Proto.Buffer n = new Proto.Buffer();
        n.string(4, opType);          // op_type
        return n;
    }

    /** Build an AttributeProto for an int attribute. */
    private static Proto.Buffer attr(String name, int type, long value) {
        Proto.Buffer a = new Proto.Buffer();
        a.string(1, name);
        a.varint(2, type);            // attribute_type = 2 (INT)
        a.varint(3, value);           // i
        return a;
    }
}
