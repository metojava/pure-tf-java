# pure-tf-java

A **pure-Java** define-by-run autograd engine and Keras-flavoured neural
network library - zero TensorFlow, zero JNI, zero native code, zero
dependencies. One small jar carries the forward math, reverse-mode autodiff,
optimizers, training loop and model persistence.

This is the native-free replacement for tensorflow-capi-java: every layer
that used to emit TF graph ops now runs on the built-in tape with hand-written
analytic gradients.

## Quick start

```
run.cmd                  # XOR + regression demos
run.cmd gradcheck        # finite-difference verification of every op
run.cmd features         # integration smoke test (MLP / attention / LSTM / CNN)
run.cmd fashion [5]      # Fashion-MNIST MLP (784-256-10)
run.cmd fashion-cnn [5] [--fast]   # Fashion-MNIST CNN, --fast = 12k subset
```

Requirements: JDK 17+, Maven. Nothing else.

## Layer zoo

| Group | Layers |
|---|---|
| Core | Dense, Dropout, Flatten, Reshape, Sequential |
| Activations | ReLU, LeakyReLU, ELU, SELU, SiLU, GELU, Softplus, Sigmoid, Tanh, Softmax |
| Normalization | LayerNorm1D, BatchNorm1D, BatchNorm2D |
| Vision | Conv2D (im2col), MaxPool2D, AvgPool2D, ZeroPad2D, Upsample2D |
| Sequence | Embedding, PositionalEncoding, SelfAttention, MultiHeadAttention (causal opt.), RNN, LSTM, GRU |

Every layer's gradient is verified numerically by `run.cmd gradcheck`
(30 checks, central differences).

## Keras-style trainer + persistence

```java
Sequential net = new Sequential()
        .add(new Conv2D(8, 3, 3, 1, 1, Padding.SAME)).add(new ReLU())
        .add(new MaxPool2D(2, 2, 2, 2, Padding.VALID))
        .add(new Flatten())
        .add(new Dense(128, Dense.Act.RELU))
        .add(new Dense(10));

History h = net.fitHistory(trainX, labels, 10,
        30 /*epochs*/, 96 /*batch*/,
        new Optim.Adam(0.0015f),
        LrSchedule.cosine(0.0015f, 0.0002f, 30),
        0.1f,                                    // validation split
        new EarlyStopping(Monitor.LOSS, 6, 0),   // patience 6
        true);
// Epoch k/N - loss .. - accuracy .. - val_loss .. - val_acc ..

net.saveTo(Path.of("model.bin"));                 // PTF1 binary format
Sequential restored = Sequential.loadFrom(Path.of("model.bin"));
TrainingCurves.save("curves", h, "target/curves.png");   // loss/acc PNG
```

`Optim.Stepper` implementations: `Adam`, `Momentum`, `RMSProp` (+ static
`sgd`). `LrSchedule`: `constant`, `step`, `cosine`, `exponential`, or a lambda.

## Low-level tape API

The layers compile down to the raw autograd tape, which remains fully
accessible:

```java
Tensor w1 = Tensor.glorot(rnd, 2, 8, 2, 8);      // trainable param
Tensor b1 = Tensor.param(new float[8], 8);

for (int step = 0; step < steps; step++) {
    Tensor.resetGraph();                          // fresh tape
    var x  = Tensor.of(batchX, B, 2);             // input leaf
    var loss = x.matmul(w1).addBias(b1).tanh().matmul(w2).mse(batchY);
    for (Tensor p : params) p.zeroGrad();
    loss.backward();                              // full backprop
    adam.step(params);
}
```

### Ops (each with analytic gradients)

- Linear algebra: `matmul`, `bmm` (batched), `addBias`, broadcast `broadcastAdd` /
  `broadcastMul` (NumPy right-aligned rules)
- Shape: `reshape`, `permute`, `sliceLast`, `rows`, `concatLast`,
  `meanAxis` / `sumAxis`
- Elementwise: `mul`, `scale`, `addScalar`, `power`
- Activations: `relu`, `leakyRelu`, `elu`, `selu`, `silu`, `gelu`, `softplus`,
  `sigmoid`, `tanh`, row-`softmax`
- Gather/vision: `embedding`, `pad2d`, `im2col` (col2im backward),
  `maxPool2d` (argmax routing), `avgPool2d`, `upsample2d`
- Losses: `mse`, fused `softmaxCE`

## Architecture notes

- Tape-based define-by-run autograd (micrograd lineage): each op records a
  backprop closure; `backward()` walks creation order in reverse.
- Conv2D is im2col + the tested rank-2 matmul, so its correctness rides on
  already-verified primitives; pooling ops route gradients via argmax masks.
- RNN/LSTM/GRU unroll through primitive ops - BPTT falls out of the tape for
  free.
- Model format `PTF1`: layer type codes + spec args + raw parameter tensors +
  extra state (BatchNorm running stats).
- Max-pool is piecewise linear; its gradient check uses well-separated window
  maxima (finite differences are meaningless at ties).

## Parity status vs tensorflow-capi-java

Done: optimizers (SGD/Momentum/RMSProp/Adam), all activation layers, norm
layers, embedding + positional encoding, attention stack, recurrent stack
(RNN/LSTM/GRU), conv/pool/pad/upsample vision stack, early stopping, LR
schedules, validation split, model save/load, evaluation reports.

Not ported (by design): CrossAttention layer (needs two-input flows),
DepthwiseConv2D, SavedModel import/export (no native runtime to talk to),
GPU execution.

Performance expectation: showcase-sized models train comfortably on CPU
(Fashion-MNIST MLP ~30 s/epoch, CNN subset epoch ~25 s). A JDK Vector API pass
on matmul/im2col is the obvious next speedup.
