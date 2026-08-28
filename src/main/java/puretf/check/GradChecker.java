package puretf.check;

import puretf.Tensor;

import java.util.Random;
import java.util.function.Function;

/**
 * Reusable numerical gradient verifier. Given a pure function that builds a
 * scalar loss from parameter tensors, it compares the analytic gradients
 * produced by {@code Tensor#backward()} against central finite differences.
 *
 * <p>Use this to verify a custom {@code Layer} or custom {@code Tensor} op
 * end to end, e.g. after writing your own layer's {@code apply}: slice out its
 * weight tensors, build the loss, and call {@link #isCorrect}.</p>
 *
 * <pre>{@code
 * // verify a custom layer's analytic gradients once
 * boolean ok = GradChecker.isCorrect("MyLayer",
 *         ps -> ps[0].matmul(ps[1]).addBias(ps[2]).silu().sumAxis(0).sumAxis(0),
 *         GradChecker.randn(8, 5),   // input features
 *         GradChecker.randn(5, 3),   // weight W
 *         GradChecker.randn(3));     // bias b
 * }</pre>
 */
public final class GradChecker {

    /** Finite-difference step. */
    public static final float H = 1e-2f;
    /** Relative-gap threshold under which a check passes. */
    public static final double TOL = 1e-2f;
    private static final Random RND = new Random(42L);

    private GradChecker() {}

    /**
     * Verifies the analytic gradients of {@code build} against finite
     * differences for every element of every given parameter.
     *
     * @return true if {@code maxError(...) < TOL}, printing a status line.
     */
    public static boolean check(String name,
                                Function<Tensor[], Tensor> build,
                                Tensor... seeds) {
        double err = maxError(build, seeds);
        boolean ok = err < TOL;
        System.out.printf("%-28s max|analytic-numeric| = %.6f  %s%n",
                name, err, ok ? "ok" : "FAIL");
        return ok;
    }

    /**
     * Like {@link #check} but throws {@link AssertionError} on failure.
     * Convenient inside test method bodies.
     */
    public static void verify(String name,
                              Function<Tensor[], Tensor> build,
                              Tensor... seeds) {
        if (!check(name, build, seeds))
            throw new AssertionError("gradient check failed: " + name);
    }

    /**
     * Returns the largest absolute gap between the analytic gradient and the
     * numeric (central-difference) gradient across all inputs.
     */
    public static double maxError(Function<Tensor[], Tensor> build,
                                  Tensor... seeds) {
        Tensor[] ps = new Tensor[seeds.length];
        for (int i = 0; i < seeds.length; i++)
            ps[i] = Tensor.param(seeds[i].data.clone(), seeds[i].shape);

        Tensor.resetGraph();
        Tensor loss = sumAll(build.apply(ps));
        loss.backward();

        double maxErr = 0;
        for (Tensor p : ps) {
            if (p.grad == null) continue;
            float[] analytic = p.grad;
            for (int i = 0; i < p.data.length; i++) {
                float orig = p.data[i];
                p.data[i] = orig + H;
                Tensor.resetGraph();
                double lp = sumAll(build.apply(ps)).data[0];
                p.data[i] = orig - H;
                Tensor.resetGraph();
                double lm = sumAll(build.apply(ps)).data[0];
                p.data[i] = orig;
                double numeric = (lp - lm) / (2.0 * H);
                maxErr = Math.max(maxErr, Math.abs(numeric - analytic[i]));
            }
        }
        return maxErr;
    }

    /** Reduces a tensor to a 1-element scalar by sum-axes. */
    public static Tensor sumAll(Tensor t) {
        while (t.shape.length > 0) t = t.sumAxis(0);
        return t;
    }

    // ---------------------- random inputs ---------------------------------

    public static Tensor randn(int... shape) {
        return Tensor.randn(RND, 1f, shape);
    }

    public static Tensor randPositive(int... shape) {
        float[] d = new float[(int) product(shape)];
        for (int i = 0; i < d.length; i++) d[i] = 0.5f + RND.nextFloat();
        return Tensor.of(d, shape);
    }

    private static long product(int... shape) {
        long p = 1;
        for (int s : shape) p *= s;
        return p;
    }
}
