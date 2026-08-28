package puretf.check;

import puretf.Tensor;

import java.util.Random;
import java.util.function.Function;

/**
 * Numerical gradient verification for every autograd op: analytic gradients
 * from backward() are compared against central finite differences.
 * Run: run.cmd gradcheck
 */
public final class GradCheck {

    private static final Random RND = new Random(42L);

    public static void main(String[] args) {
        // ---- existing core ops ------------------------------------------
        check("matmul", ps -> ps[0].matmul(ps[1]),
                randn(2, 3), randn(3, 4));
        check("broadcastAdd [B,N]+[N]", ps -> ps[0].broadcastAdd(ps[1]),
                randn(3, 4), randn(4));
        check("mul", ps -> ps[0].mul(ps[1]),
                randn(2, 3), randn(2, 3));
        int[] labels = {0, 3, 1, 2};
        check("softmaxCE", ps -> Tensor.softmaxCE(ps[0], labels),
                randn(4, 5));

        // ---- shape & broadcasting ---------------------------------------
        check("reshape", ps -> ps[0].reshape(3, 2),
                randn(2, 3));
        check("permute", ps -> ps[0].permute(0, 2, 1),
                randn(2, 3, 2));
        check("broadcastAdd [B,T,D]+[D]", ps -> ps[0].broadcastAdd(ps[1]),
                randn(2, 2, 3), randn(3));
        check("meanAxis", ps -> ps[0].meanAxis(1),
                randn(3, 4));
        check("sumAxis", ps -> ps[0].sumAxis(0),
                randn(3, 4));
        check("meanAxis 3D", ps -> ps[0].meanAxis(2),
                randn(2, 3, 4));
        check("broadcastMul [B,T,D]*[D]", ps -> ps[0].broadcastMul(ps[1]),
                randn(2, 2, 3), randn(3));
        check("broadcastMul [B,N]*[N]", ps -> ps[0].broadcastMul(ps[1]),
                randn(2, 3), randn(3));

        // ---- scalar elementwise -----------------------------------------
        check("scale", ps -> ps[0].scale(-1.5f),
                randn(2, 3));
        check("addScalar", ps -> ps[0].addScalar(0.7f),
                randn(2, 3));
        check("power +0.5", ps -> ps[0].power(0.5f),
                randPositive(2, 3));
        check("power -0.5", ps -> ps[0].power(-0.5f),
                randPositive(2, 3));

        // ---- activations -------------------------------------------------
        check("leakyRelu", ps -> ps[0].leakyRelu(0.1f), randn(3, 3));
        check("elu", ps -> ps[0].elu(1f), randn(3, 3));
        check("selu", ps -> ps[0].selu(), randn(3, 3));
        check("silu", ps -> ps[0].silu(), randn(3, 3));
        check("gelu", ps -> ps[0].gelu(), randn(3, 3));
        check("softplus", ps -> ps[0].softplus(), randn(3, 3));
        float[] tgt = new float[12];
        for (int i = 0; i < tgt.length; i++) tgt[i] = RND.nextFloat();
        check("softmax", ps -> ps[0].softmax().mul(Tensor.of(tgt.clone(), 3, 4)),
                randn(3, 4));

        // ---- gather / batched matmul -------------------------------------
        int[] ids = {1, 3, 1};
        check("embedding", ps -> Tensor.embedding(ps[0], ids),
                randn(5, 4));
        check("bmm", ps -> ps[0].bmm(ps[1]),
                randn(2, 3, 4), randn(2, 4, 5));
        check("sliceLast", ps -> ps[0].sliceLast(1, 2),
                randn(2, 5, 4));
        check("rows", ps -> ps[0].rows(1, 2),
                randn(4, 3));

        // ---- vision ops (NHWC) -------------------------------------------
        check("pad2d", ps -> ps[0].pad2d(1, 1, 2, 0),
                randn(2, 3, 3, 3));
        // conv via im2col + matmul (weights reshaped [K,F])
        check("conv im2col", ps -> {
                    int b = 2, oh = 3, ow = 3, K = 18, F = 4;
                    Tensor cols = ps[0].im2col(3, 3, 1, 1, 0, 0, oh, ow);
                    return cols.matmul(ps[1]).reshape(b, oh, ow, F);
                },
                randn(2, 5, 5, 2), randn(18, 4));
        check("avgPool2d", ps -> ps[0].avgPool2d(2, 2, 2, 2, 0, 0, 2, 2),
                randn(2, 4, 4, 3));
        // max-pool is piecewise linear: needs well-separated window maxima
        // for finite differences to agree
        check("maxPool2d", ps -> ps[0].maxPool2d(2, 2, 2, 2, 0, 0, 2, 2),
                spreadMax(4, 4, 2));
        check("upsample2d", ps -> ps[0].upsample2d(2, 2),
                randn(2, 2, 2, 3));

        System.out.println("\nall gradient checks passed");
    }

    // ---------------------- machinery ------------------------------------
    // Reusable numerical verifier; the checks below just delegate to it.

    private static void check(String name, Function<Tensor[], Tensor> build,
                              Tensor... seeds) {
        GradChecker.verify(name, build, seeds);
    }

    private static Tensor sumAll(Tensor t) { return GradChecker.sumAll(t); }

    private static Tensor randn(int... shape) { return GradChecker.randn(shape); }

    private static Tensor randPositive(int... shape) {
        return GradChecker.randPositive(shape);
    }

    /** NHWC [1,H,W,C] where every kxk window has a unique, far-apart max. */
    private static Tensor spreadMax(int h, int w, int c) {
        float[] d = new float[h * w * c];
        int block = 0;
        for (int by = 0; by < h / 2; by++)
            for (int bx = 0; bx < w / 2; bx++)
                for (int ch = 0; ch < c; ch++) {
                    float base = 0.2f * (block % 11);
                    int[][] offs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
                    int rot = block % 4;
                    d[((by * 2 + offs[rot][0]) * w
                            + bx * 2 + offs[rot][1]) * c + ch] = base + 8;
                    d[((by * 2 + offs[(rot + 1) % 4][0]) * w
                            + bx * 2 + offs[(rot + 1) % 4][1]) * c + ch] = base + 0.6f;
                    d[((by * 2 + offs[(rot + 2) % 4][0]) * w
                            + bx * 2 + offs[(rot + 2) % 4][1]) * c + ch] = base - 1.4f;
                    d[((by * 2 + offs[(rot + 3) % 4][0]) * w
                            + bx * 2 + offs[(rot + 3) % 4][1]) * c + ch] = base - 4.2f;
                    block++;
                }
        return Tensor.of(d, 1, h, w, c);
    }

    private GradCheck() {}
}
