package puretf.nn;

import puretf.Tensor;

/** Softmax over the last axis (use before a plain MSE-style loss only;
 *  classifiers should end in raw logits + fitClassifier). */
public final class Softmax extends ActivationLayer {

    @Override
    protected Tensor apply(Tensor x) { return x.softmax(); }

    @Override public int typeCode() { return 13; }
}
