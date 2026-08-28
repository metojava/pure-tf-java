package puretf.data;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Simple, dependency-free tokenizer for character-level NLP, sized to power
 * the {@code LSTM}/{@code GRU}/{@code RNN} + {@code Embedding} demos. Builds a
 * vocabulary from text, then converts between strings and integer token ids.
 *
 * <p>Modeled on the classic "min-char-rnn" style character language model:
 * one token per character. Word-tokenizing is a layer you can add on top by
 * feeding already-split tokens to {@link #encodeTokens}.</p>
 *
 * <pre>{@code
 * String corpus = "hello world hello world";
 * Tokenizer t = new Tokenizer(corpus);          // learns the char vocab
 * int[] ids = t.encode("hello");                // [.., .., .., .., ..]
 * String s = t.decode(new int[]{t.encode("hello")[0], 2});  // back to text
 * }</pre>
 */
public final class Tokenizer {

    /** Special id for the end-of-sequence marker if present in the corpus. */
    public static final int EOS = 2;

    private final List<String> vocab;                 // id -> token
    private final Map<String, Integer> table;         // token -> id

    /** Builds a character vocabulary from the whole {@code text}. */
    public Tokenizer(String text) {
        LinkedHashMap<String, Integer> seen = new LinkedHashMap<>();
        for (char c : text.toCharArray())
            seen.putIfAbsent(String.valueOf(c), seen.size());
        vocab = new ArrayList<>(seen.keySet());
        table = new LinkedHashMap<>();
        for (int i = 0; i < vocab.size(); i++) table.put(vocab.get(i), i);
    }

    /** Vocab size (= embedding table rows). */
    public int size() { return vocab.size(); }

    /** Whether {@code token} is known to this vocabulary. */
    public boolean has(String token) { return table.containsKey(token); }

    public int id(String token) {
        Integer i = table.get(token);
        if (i == null) throw new IllegalArgumentException("token not in vocab: " + token);
        return i;
    }

    public String token(int id) { return vocab.get(id); }

    /** Encodes a whole string to a sequence of character ids. */
    public int[] encode(String text) {
        int n = text.length();
        int[] ids = new int[n];
        for (int i = 0; i < n; i++) ids[i] = id(String.valueOf(text.charAt(i)));
        return ids;
    }

    /** Encodes an arbitrary sequence of pre-split tokens. */
    public int[] encodeTokens(String[] tokens) {
        int[] ids = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) ids[i] = id(tokens[i]);
        return ids;
    }

    /** Decodes ids back to text. */
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) sb.append(vocab.get(id));
        return sb.toString();
    }

    /** All ids as a readable string, useful for debugging. */
    public String toString(Iterable<Integer> ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) sb.append(id).append(' ');
        return sb.toString().trim();
    }
}
