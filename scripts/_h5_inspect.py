import h5py
p = r"C:\Users\dell\AppData\Local\Temp\java_test_model.weights.h5"
def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        print("DS", name, obj.shape, obj.dtype, "storage=%s" % ("contiguous" if obj.compression is None else obj.compression))
        print("  byteorder raw offset:", obj.id.get_offset(), "len", obj.id.get_storage_size())
with h5py.File(p, "r") as f:
    f.visit(lambda n: walk(n, f[n]))
