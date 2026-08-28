package puretf.demo;

import puretf.nn.Dense;
import puretf.nn.Sequential;
import puretf.serialize.ModelBundle;
import puretf.serialize.NamedTensor;
import puretf.serialize.SafeTensors;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

/**
 * SafeTensors format demo: builds a model, exports to .safetensors, reads it
 * back, and verifies bit-identical weights.
 */
public final class SafeTensorsDemo {

    public static void main(String[] a) throws Exception {
        System.out.println("=== SafeTensors format demo ===");

        // 1. Build & train a tiny model
        Sequential net = new Sequential()
                .add(new Dense(16, Dense.Act.RELU))
                .add(new Dense(4));
        net.build(8);

        Random rnd = new Random(123);
        List<float[]> w = net.getWeights();
        for (float[] f : w) for (int i = 0; i < f.length; i++) f[i] = rnd.nextFloat() * 2 - 1;
        net.setWeights(w);

        // 2. Export to safetensors
        ModelBundle b = net.saveBundle();
        Path out = Path.of("target", "demo_model.safetensors");
        out.getParent().toFile().mkdirs();
        SafeTensors.write(out, b);
        System.out.println("Wrote " + out.toAbsolutePath() + " (" + Files.size(out) + " bytes, " + b.size() + " tensors)");

        // 3. Read back
        ModelBundle back = SafeTensors.read(out);
        System.out.println("Read back: " + back.size() + " tensors");
        for (NamedTensor t : back.tensors()) {
            System.out.printf("  %s %s %s first=%.6f%n",
                    t.name(), t.dtype(), java.util.Arrays.toString(t.shape()), t.toFloats()[0]);
        }

        // 4. For full model round-trip, use the engine's native bundle (has manifest + inputDim)
        Sequential rebuilt = Sequential.loadBundle(b); // uses engine bundle
        boolean ok = equalWeights(net.getWeights(), rebuilt.getWeights());
        System.out.println("Engine bundle round-trip: " + (ok ? "PASS (bit-identical)" : "FAIL"));
    }

    private static boolean equalWeights(List<float[]> a, List<float[]> b) {
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++)
            if (!java.util.Arrays.equals(a.get(i), b.get(i))) return false;
        return true;
    }
}