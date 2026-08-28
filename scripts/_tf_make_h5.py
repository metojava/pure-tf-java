import os, tempfile, numpy as np, tensorflow as tf
tf.keras.utils.set_random_seed(0)
m = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(3,), name="dense_1", use_bias=False),
    tf.keras.layers.Dense(2, name="dense_2", use_bias=True),
])
m.build((None, 3))
h5 = os.path.join(tempfile.gettempdir(), "java_test_model.weights.h5")
if os.path.exists(h5): os.remove(h5)
m.save_weights(h5)  # HDF5 (Keras weights)
print("h5:", h5, os.path.getsize(h5))
print("names/shapes:")
for layer in m.layers:
    for w in layer.weights:
        print("  ", w.name, w.numpy().shape, w.numpy().dtype)
