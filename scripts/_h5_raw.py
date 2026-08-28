import h5py
p = r"C:\Users\dell\AppData\Local\Temp\java_test_model.weights.h5"
with h5py.File(p, "r", driver="file") as f:
    fa = f.id.get_access_plist()
    fi = f.id.get_create_plist()
    import struct
    print("file", f.filename)
    print("root group addr:", hex(f.id.get_group_handle()[0]) if False else "n/a")
# Use h5py low-level to print superblock
import h5py
fid = h5py.File(p, "r")
sb = fid.id.get_superblock() if hasattr(fid.id, "get_superblock") else None
try:
    print("superblock:", sb)
except Exception as e:
    print("sb err", e)
print("attrs root:", list(fid.attrs.keys()))
# print the low-level object header addresses
import os
sz = os.path.getsize(p)
print("filesize", sz)
fid.close()
# raw parse of superblock
with open(p, "rb") as fin:
    data = fin.read()
print("signature ok:", data[:8] == b"\x89HDF\r\n\x1a\n")
print(" superblock version", data[8])
print(" free space version", data[9], "root group symbol table entry version", data[10])
print("reserved?", data[11], "shared header msg version", data[12])
print("sizeof offsets", data[13], "sizeof lengths", data[14])
print("group leaf node K", data[15], "reserved?", data[16])
print("consistency flags", data[17])
print("indexed storage internal", struct.unpack("<q", data[18:30] if data[13]==8 else data[18:26])[0] if False else "")
# version >=2 superblock: after sizeof off/len (2 bytes) there's a 8-byte root group version? 
print(" bytes 8..18:", " ".join("%02x" % b for b in data[8:18]))
print(" bytes 18..48:", " ".join("%02x" % b for b in data[18:48]))
import struct
print("v1 root group symbol table (bytes). version scheme depends on data[8].")
print("v1/v2 guess root addr at [18:26] (8 bytes) =", hex(struct.unpack("<Q", data[18:26])[0]) if data[13]==8 else struct.unpack("<I", data[18:22])[0])
# superblock v2: offset size 8 (data[13]) -> root group object header address is at bytes 18..26 (8 bytes) for v2
import struct
if data[8] == 2:
    off = 18
    root_addr = struct.unpack("<Q", data[off:off+8])[0]
    print("v2 root group obj header addr:", hex(root_addr))
    # next data: superblock extension addr (8)
    ext = struct.unpack("<Q", data[off+8:off+16])[0]
    print("v2 superblock extension addr:", hex(ext))
    # then: file free-space addr (8), end-of-address space (8), driver addr (8),
    # number of root group symbol table entries... actually v2: after ext addr comes
    # "total# filespace"?? Let's just print 64 bytes from root_addr
    print("bytes at root_addr:", " ".join("%02x" % x for x in data[root_addr:root_addr+64]))
    print("bytes at root_addr+8:", " ".join("%02x" % x for x in data[root_addr+8:root_addr+40]))
