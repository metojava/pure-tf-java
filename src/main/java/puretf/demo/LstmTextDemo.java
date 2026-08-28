package puretf.demo;

import puretf.Optim;
import puretf.Tensor;
import puretf.data.Sampler;
import puretf.data.Tokenizer;
import puretf.nn.Dense;
import puretf.nn.Embedding;
import puretf.nn.LSTM;
import puretf.nn.Sequential;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Character-level language model with an {@link LSTM}, pure Java.
 * Teaches a tiny recurrent net to reproduce a short repeating phrase using
 * teacher forcing, then generates new text autoregressively with
 * {@link Sampler}.
 *
 * <p>The recurrent body (Embedding -&gt; LSTM emitting every timestep) is a
 * separate {@link Sequential} from the classifier head (Dense), because
 * {@code Dense} operates on {@code [batch, features]} whereas the LSTM emits
 * {@code [1, T, units]}; we reshape to {@code [T, units]} before the head.</p>
 */
public final class LstmTextDemo {

    public static void main(String[] args) {
        int reps = 40;
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < reps; i++) sb.append("to be or not to be, that is the question. ");
        String corpus = sb.toString();

        Tokenizer tok = new Tokenizer(corpus);
        int vocab = tok.size();
        int[] data = tok.encode(corpus);
        int T = data.length;

        int embedDim = 32, lstm = 80;
        Sequential body = new Sequential()
                .add(new Embedding(vocab, embedDim))
                .add(new LSTM(lstm, true));            // [1, T, lstm]
        Sequential head = new Sequential().add(new Dense(vocab));  // [features] -> vocab
        body.build(T);
        head.build(lstm);

        List<Tensor> params = new ArrayList<>();
        params.addAll(body.params());
        params.addAll(head.params());
        Optim.Adam adam = new Optim.Adam(0.004f);

        // target[ i ] = char after position i (wraps at the end)
        int[] target = new int[T];
        for (int i = 0; i < T - 1; i++) target[i] = data[i + 1];
        target[T - 1] = data[0];

        int steps = 300;
        for (int s = 0; s < steps; s++) {
            Tensor.resetGraph();
            // [1,T] ids -> body -> [1,T,lstm] -> reshape to [T,lstm] -> head [T,vocab]
            Tensor bodyOut = body.forward(Tensor.of(floats(data), 1, T));
            Tensor logits = head.forward(bodyOut.reshape(T, lstm));
            Tensor loss = Tensor.softmaxCE(logits, target);
            for (Tensor p : params) p.zeroGrad();
            loss.backward();
            body.clipGradNorm(1f);   // global-ish clip over both halves
            head.clipGradNorm(1f);
            adam.step(params);
            if (s % 50 == 0 || s == steps - 1)
                System.out.printf("step %4d  loss %.4f%n", s, loss.data[0]);
        }

        System.out.println("\ngenerated (temperature 0.4):");
        System.out.println("  " + generate(body, head, tok, "to be", 80, 0.4f, new Random(3L)));
        System.out.println("generated (temperature 1.0):");
        System.out.println("  " + generate(body, head, tok, "to be", 80, 1.0f, new Random(3L)));
    }

    private static String generate(Sequential body, Sequential head, Tokenizer tok,
                                   String seed, int len, float temp, Random rnd) {
        int[] ids = tok.encode(seed);
        StringBuilder out = new StringBuilder(seed);
        for (int i = 0; i < len; i++) {
            Tensor.resetGraph();
            // [1,cur] ids -> body -> [1,cur,U]; take the LAST timestep row
            Tensor bodyOut = body.forward(Tensor.of(floats(ids), 1, ids.length));
            int U = bodyOut.shape[2];
            float[] lastRow = new float[U];
            System.arraycopy(bodyOut.data, (ids.length - 1) * U, lastRow, 0, U);
            Tensor logits = head.forward(Tensor.of(lastRow, 1, U));   // [1, V]
            int next = Sampler.categorical(rnd, logits.data, temp);
            out.append(tok.token(next));
            int[] newIds = new int[ids.length + 1];
            System.arraycopy(ids, 0, newIds, 0, ids.length);
            newIds[ids.length] = next;
            ids = newIds;
        }
        return out.toString();
    }

    private static float[] floats(int[] a) {
        float[] o = new float[a.length];
        for (int i = 0; i < a.length; i++) o[i] = a[i];
        return o;
    }
}
