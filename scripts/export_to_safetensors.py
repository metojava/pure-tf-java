#!/usr/bin/env python3
"""
export_to_safetensors.py
-----------------------
Export a model's weights to a single .safetensors file that the pure-tf-java
Java engine can read with `puretf.serialize.SafeTensors.read(path)`.

Usage
-----
  # PyTorch (.pt/.pth)  -- requires `torch`
  python scripts/export_to_safetensors.py --torch model.pth \
          --out model.safetensors \
          --rename layer_0/kernel=features.0.weight,layer_0/bias=features.0.bias

  # TensorFlow / Keras (.h5 or SavedModel dir)  -- requires `tensorflow`
  python scripts/export_to_safetensors.py --tf model.h5 \
          --out model.safetensors \
          --rename dense_1/kernel=trained/dense_1/Kernel:0

  # ONNX (.onnx)  -- requires `onnx`
  python scripts/export_to_safetensors.py --onnx model.onnx --out model.safetensors

  # NumPy .npz
  python scripts/export_to_safetensors.py --npz weights.npz --out model.safetensors

The --rename flag lets you map arbitrary external parameter names onto the
engine's `layer_<i>/<role>` convention used by `Sequential.saveBundle`, so the
result can be combined with the engine's manifest for a full reload.
"""

import argparse, os, sys

def _to_safetensors(state, out):
    import numpy as np
    import json, struct
    header = {}
    body = bytearray()
    for name in list(state.keys()):
        a = np.ascontiguousarray(state[name])
        dt = str(a.dtype)
        stype = {
            "float32": "F32", "float16": "F16", "float64": "F64",
            "int32": "I32", "int64": "I64", "int16": "I16",
            "int8": "I8", "uint8": "U8", "bool": "BOOL",
        }.get(dt, "F32")
        b = a.view(np.uint8).tobytes()
        pad = (-len(body)) % 8
        body += b"\x00" * pad
        start = len(body)
        body += b
        end = len(body)
        header[name] = {"dtype": stype, "shape": list(a.shape), "data_offsets": [start, end]}
    header_b = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(header_b)))
        f.write(header_b)
        f.write(body)
    print("Wrote", out, "with", len(state), "tensors")

def _rename(state, mapping):
    out = dict(state)
    for pair in mapping:
        dst, src = pair.split("=", 1)
        if src in out:
            out[dst] = out.pop(src)
        else:
            sys.stderr.write("WARN: rename target '%s' not found\n" % src)
    return out

def from_torch(path, renames):
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    state = {k: v.detach().cpu().numpy() for k, v in sd.items()}
    return _rename(state, renames)

def from_tf(path, renames):
    # Try a full Keras model first, then fall back to a weights-only .h5 via h5py.
    try:
        import tensorflow as tf
        m = tf.keras.models.load_model(path)
        state = {}
        for layer in m.layers:
            for w in layer.weights:
                state[w.name] = w.numpy()
        return _rename(state, renames)
    except Exception:
        pass
    import h5py, numpy as np
    state = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                state[name] = np.asarray(obj)
        f.visititems(visit)
    return _rename(state, renames)

def from_onnx(path, renames):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)
    state = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}
    return _rename(state, renames)

def from_npz(path, renames):
    import numpy as np
    d = np.load(path)
    state = {k: d[k] for k in d.files}
    return _rename(state, renames)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch", "--pth", dest="torch")
    ap.add_argument("--tf", "--h5", dest="tf")
    ap.add_argument("--onnx")
    ap.add_argument("--npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rename", nargs="*", default=[])
    args = ap.parse_args()
    state = None
    if args.torch: state = from_torch(args.torch, args.rename)
    if args.tf:    state = from_tf(args.tf, args.rename)
    if args.onnx:  state = from_onnx(args.onnx, args.rename)
    if args.npz:   state = from_npz(args.npz, args.rename)
    if state is None: ap.error("must supply one of --torch/--tf/--onnx/--npz")
    _to_safetensors(state, args.out)

if __name__ == "__main__":
    main()
