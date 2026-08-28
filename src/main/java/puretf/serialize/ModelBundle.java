package puretf.serialize;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A framework-agnostic model representation: an ordered list of named
 * parameter/buffer tensors plus an optional layer manifest describing the
 * network topology. This is the hub that all import/export drivers plug into.
 *
 * <ul>
 *   <li><b>Engine &rarr; Bundle</b>: {@link #fromSequential} snapshots every
 *       trained parameter of a {@code Sequential} with a stable, layernames
 *       convention ({@code dense_0/kernel}, {@code lstm_2/recurrent}, ...).</li>
 *   <li><b>Bundle &rarr; Engine</b>: {@link #toSequential} rehydrates a
 *       compatible {@code Sequential} from those named tensors.</li>
 *   <li><b>Bundle &harr; external</b>: the format drivers translate the same
 *       tensors to/from safetensors, ONNX, PyTorch and TensorFlow files.</li>
 * </ul>
 *
 * <p>The bundle keeps an arbitrary {@code metadata} map so format-specific
 * readers can stash things like the optimizer state or the source shape.</p>
 */
public final class ModelBundle {

    private final List<NamedTensor> tensors = new ArrayList<>();
    private final Map<String, String> metadata = new LinkedHashMap<>();

    public List<NamedTensor> tensors() { return tensors; }
    public Map<String, String> metadata() { return metadata; }

    public ModelBundle add(NamedTensor t) {
        tensors.add(t);
        return this;
    }

    public int size() { return tensors.size(); }

    /** The i-th tensor in insertion order. */
    public NamedTensor paramByIndex(int i) { return tensors.get(i); }

    /** Look up a tensor by exact name. */
    public NamedTensor find(String name) {
        for (NamedTensor t : tensors) if (t.name().equals(name)) return t;
        return null;
    }

    /** Look up by a prefix (first match whose name starts with {@code prefix}). */
    public NamedTensor findPrefix(String prefix) {
        for (NamedTensor t : tensors) if (t.name().startsWith(prefix)) return t;
        return null;
    }

    public static ModelBundle of(List<NamedTensor> ts) {
        ModelBundle b = new ModelBundle();
        for (NamedTensor t : ts) b.add(t);
        return b;
    }

    public List<LayerSpec> layerManifest() { return layerSpecs; }

    private final List<LayerSpec> layerSpecs = new ArrayList<>();

    /** A serializable description of one network layer (type + constructor args). */
    public static final class LayerSpec {
        public final String type;      // e.g. "Dense", "Conv2D", "LSTM"
        public final int code;         // engine layer typeCode (rebuild key)
        public final int[] args;       // constructor args as ints
        public LayerSpec(String type, int code, int[] args) {
            this.type = type; this.code = code; this.args = args;
        }
        @Override public String toString() { return type; }
    }

    public ModelBundle recordLayer(String type, int code, int[] args) {
        layerSpecs.add(new LayerSpec(type, code, args));
        return this;
    }

    public static String tensorName(int layerIndex, String role) {
        return "layer_" + layerIndex + "/" + role;
    }
}
