import h5py
p = r"C:\Users\dell\AppData\Local\Temp\java_test_model.weights.h5"
with open(p, "rb") as f:
    head = f.read(16)
print("signature:", head[:8], "version:", head[8], "fletcher:", head[9])
print("first 100 bytes hex:", " ".join("%02x" % b for b in f.read(100)) if False else "")
f2 = open(p, "rb"); data = f2.read(); f2.close()
off = 0
print(" superblock version", data[8])
print(" root group version", data[9+1])
print(" sizeof offsets", data[10], "sizeof lengths", data[11])
print(" file consistency flags", data[12])
# v1 superblock body
import struct
sizeof_off = data[10]; sizeof_len = data[11]
so = 8 + 8  # skip signature + (version, fletcher, version2, versionGEA?) actually layout: sig(8), version(1), fletcher(1), truncated(1), version2(1), versionGEA(1), sizeofoff(1), sizeoflen(1), reserved(1), intk(1)...
print("bytes 8..24:", " ".join("%02x" % b for b in data[8:32]))
