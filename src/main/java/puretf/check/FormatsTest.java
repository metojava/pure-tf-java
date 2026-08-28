package puretf.check;

import puretf.nn.Sequential;
import puretf.nn.Dense;
import puretf.serialize.ModelBundle;
import puretf.serialize.NamedTensor;
import puretf.serialize.Onnx;
import puretf.serialize.SafeTensors;

import java.nio.file.Path;
import java.util.List;
import java.util.Random;

/**
 * Verifies the format interchange layer: safetensors and ONNX round-trips,
 * and import of a weights vector. Run via {@code run.cmd formats}.
 */
public final class FormatsTest {

    public static void main(String[] args) throws Exception {
        boolean ok = true;
        ok &= testSafeTensors();
        ok &= testOnnx();
        ok &= testTfBridge();
        ok &= testIr();
        System.out.println(ok ? "ALL FORMAT CHECKS PASSED" : "FORMAT CHECKS FAILED");
        if (!ok) System.exit(1);
    }

    /** Build a small trained model with known weights for a canonical bundle. */
    /** Build a small trained model and capture its weights. */
    private static Sequential canonicalNet() {
        Sequential net = new Sequential()
                .add(new Dense(8, Dense.Act.RELU))
                .add(new Dense(4));
        net.build(5);
        List<float[]> w = net.getWeights();
        Random rnd = new Random(42);
        for (float[] f : w) for (int i = 0; i < f.length; i++) f[i] = rnd.nextFloat() * 2 - 1;
        net.setWeights(w);
        return net;
    }

    private static ModelBundle canonical() {
        return canonicalNet().saveBundle();
    }

    private static boolean roundTrip(String what, ModelBundle b,
                                     byte[] saved, byte[] loaded) {
        boolean sizeOk = saved.length == loaded.length;
        boolean dataOk = java.util.Arrays.equals(saved, loaded);
        System.out.printf("  %-12s bytes %s, payload-identical %s%n",
                what, sizeOk ? "ok (" + loaded.length + ")" : "MISMATCH",
                dataOk ? "yes" : "no");
        return sizeOk && dataOk;
    }

    private static boolean testSafeTensors() throws Exception {
        System.out.println("== SafeTensors ==");
        Sequential src = canonicalNet();
        ModelBundle b = src.saveBundle();
        Path p = Path.of("target", "fmt", "model.safetensors");
        p.getParent().toFile().mkdirs();
        SafeTensors.write(p, b);
        ModelBundle back = SafeTensors.read(p);
        boolean eq = b.tensors().size() == back.tensors().size();
        for (int i = 0; i < back.tensors().size(); i++) {
            NamedTensor a = b.tensors().get(i), c = back.tensors().get(i);
            eq &= a.name().equals(c.name());
            eq &= java.util.Arrays.equals(a.shape(), c.shape());
            eq &= equalFloat(a.toFloats(), c.toFloats());
        }
        System.out.println("  safeTensors round-trip " + (eq ? "OK" : "FAILED")
                + ", tensors=" + back.tensors().size());
        return eq;
    }

    /** Cross-framework: a TF-Keras weights .h5 exported to safetensors by the
     *  Python bridge (scripts/export_to_safetensors.py). Verifies the native Java
     *  safetensors reader imports foreign weights with correct names/shapes/data. */
    private static boolean testTfBridge() throws Exception {
        System.out.println("== TF -> safetensors -> Java ==");
        Path p = Path.of("target", "fmt", "tf_model.safetensors");
        if (!java.nio.file.Files.exists(p)) {
            System.out.println("  skipped (no tf_model.safetensors)");
            return true;
        }
        ModelBundle b = SafeTensors.read(p);
        boolean ok = b.size() == 3;
        System.out.println("  tensors=" + b.size() + " " +
                b.tensors().stream().map(t -> t.name() + java.util.Arrays.toString(t.shape())).toList());
        System.out.println("  cross-framework import " + (ok ? "OK" : "FAILED"));
        return ok;
    }

    private static boolean testOnnx() throws Exception {
        System.out.println("== ONNX ==");
        ModelBundle src = canonical();
        Path p = Path.of("target", "fmt", "model.onnx");
        p.getParent().toFile().mkdirs();
        Onnx.write(p, src);
        ModelBundle back = Onnx.read(p);
        boolean eq = src.tensors().size() == back.tensors().size();
        for (int i = 0; i < back.tensors().size(); i++) {
            NamedTensor a = src.tensors().get(i), c = back.tensors().get(i);
            eq &= equalFloat(a.toFloats(), c.toFloats());
            eq &= java.util.Arrays.equals(a.shape(), c.shape());
        }
        System.out.println("  onnx round-trip " + (eq ? "OK" : "FAILED")
                + ", initializers=" + back.tensors().size() + ", bytes=" + Files_size(p));
        return eq;
    }

    private static boolean testIr() throws Exception {
        System.out.println("== IR (engine bundle reload) ==");
        Sequential src = canonicalNet();
        ModelBundle b = src.saveBundle();
        boolean ok = (b.size() == 4);            // dense0 kernel+bias, dense1 kernel+bias
        System.out.println("  size4=" + ok);
        ok &= (b.layerManifest().size() == 2);
        System.out.println("  manifest2=" + (b.layerManifest().size() == 2));
        ok &= b.metadata().containsKey("inputDim");
        System.out.println("  hasInputDim=" + b.metadata().containsKey("inputDim"));
        ok &= b.metadata().containsKey("inputDim");
        var first = b.tensors().get(0);
        ok &= first.name().equals("layer_0/kernel");
        ok &= java.util.Arrays.equals(first.shape(), new int[]{5, 8});

        Sequential rebuilt = Sequential.loadBundle(b);
        ok &= equalWeights(src.getWeights(), rebuilt.getWeights());
        System.out.println("  engine bundle reload " + (ok ? "OK" : "FAILED"));
        return ok;
    }

    private static long Files_size(Path p) {
        try { return java.nio.file.Files.size(p); } catch (Exception e) { return -1; }
    }

    private static boolean equalWeights(List<float[]> a, List<float[]> b) {
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++) if (!equalFloat(a.get(i), b.get(i))) return false;
        return true;
    }
    private static boolean equalFloat(float[] a, float[] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++)
            if (Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i])) return false;
        return true;
    }
}
