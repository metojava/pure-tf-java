package puretf.nn;

import puretf.Tensor;

import java.util.List;

/** Base for parameter-free activation layers. */
public abstract class ActivationLayer extends Layer {

    @Override
    public List<Tensor> params() { return List.of(); }

    @Override
    protected void init(int fanIn) {}
}
