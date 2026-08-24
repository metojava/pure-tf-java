package puretf.nn;

import puretf.Tensor;

/** Zero-pads H/W of a NHWC tensor by {top, bottom, left, right}. */
public final class ZeroPad2D extends ActivationLayer {

    private final int pt, pb, pl, pr;

    public ZeroPad2D(int top, int bottom, int left, int right) {
        this.pt = top;
        this.pb = bottom;
        this.pl = left;
        this.pr = right;
    }

    @Override
    protected Tensor apply(Tensor x) { return x.pad2d(pt, pb, pl, pr); }

    @Override public int typeCode() { return 22; }

    @Override public int[] specArgs() { return new int[]{pt, pb, pl, pr}; }
}
