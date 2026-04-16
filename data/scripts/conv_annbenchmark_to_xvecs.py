import sys
import os
import h5py
import numpy as np 
import struct

"""
    Code appropriated from convert_ann_benchmark_datasets.py in flatnav library.

    python convert_annbenchmark_to_xvecs.py full/path/to/data/data.h5py --normalize
    or run getdata.sh
"""

def save_ivecs(filename, data):
    with open(filename, 'wb') as f:
        dim = data.shape[1]
        for vector in data:
            # Write dimensionality as 4-byte integer
            f.write(struct.pack('i', dim))
            # Write vector components as 4-byte floats
            f.write(vector.astype(np.int32).tobytes())

def save_fvecs(filename, data):
    with open(filename, 'wb') as f:
        dim = data.shape[1]
        for vector in data:
            # Write dimensionality as 4-byte integer
            f.write(struct.pack('i', dim))
            # Write vector components as 4-byte floats
            f.write(vector.astype(np.float32).tobytes())

def verify_fvecs(filename):
    with open(filename, 'rb') as f:
        vectors = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes: 
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec_bytes = f.read(4 * dim)
            vector = np.frombuffer(vec_bytes, dtype=np.float32)
            vectors.append(vector)
        arr = np.array(vectors)
        print(f"VERIFY {filename}: shape={arr.shape}, min={arr.min()}, max={arr.max()}")
        print(f"  sample: {arr[0, :5]}")  # First 5 elements of first vector
        return arr

# get filename
filename = sys.argv[1] # datapath + 

# read bool normalize from command
normalize = False
if '--normalize' in sys.argv:
    print("Normalizing") 
    normalize = True

#read file from command
f = h5py.File(filename, 'r')

# ['distances', 'neighbors', 'test', 'train']
# file format:
train = f.get('train')[()]
test = f.get('test')[()]
gtruth = f.get('neighbors')[()]

# preview data metadata:
print(train.shape,train.dtype)
print(test.shape,test.dtype)
print(gtruth.shape,gtruth.dtype)

# view data
print("before normalization:")
print(f"train before normalization % non-zero : {np.count_nonzero(train)} / {train.size} = {np.count_nonzero(train) / train.size * 100}%")
print(f"train sample from ind = 0: {train[0, 0:5]}")
print(f"train max: {train.max()}")
print(f"train max: {train.min()}")

# normalize
if normalize:
    # print(f"norm = {np.linalg.norm(train, axis = 1, keepdims = True) + 1e-30}")
    train /= ( np.linalg.norm(train, axis = 1, keepdims = True) + 1e-30 )
    test /= ( np.linalg.norm(test, axis = 1,keepdims = True) + 1e-30)

# view data
print("after normalization:")
print(f"train before normalization % non-zero : {np.count_nonzero(train)} / {train.size} = {np.count_nonzero(train) / train.size * 100}%")
print(f"train sample from ind = 0: {train[0, 0:5]}")
print(f"train max: {train.max()}")
print(f"train max: {train.min()}")

# save to file
filename = os.path.dirname(filename)
print(filename)
save_fvecs(filename+"/data.fvecs", train)
save_fvecs(filename+"/qs.fvecs", test)
save_ivecs(filename+"/gt.ivecs", gtruth)

# print expected dimensions
print(f"Expected in C++:")
print(f"  - data.fvecs: {train.shape[0]} vectors, dim={train.shape[1]}")
print(f"  - qs.fvecs: {test.shape[0]} vectors, dim={test.shape[1]}") 
print(f"  - gt.ivecs: {gtruth.shape[0]} vectors, dim={gtruth.shape[1]}")

# After saving, verify
print("=== VERIFYING SAVED FILES ===")
data_verify = verify_fvecs(filename+"/data.fvecs")
qs_verify = verify_fvecs(filename+"/qs.fvecs")