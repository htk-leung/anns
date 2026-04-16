import sys
import os
import h5py
import numpy as np

"""
    THIS ONE IS WRONG
    Code written to convert hdf5 files to fvec/ivec files for consistent data loading.
    Note code is written in accordance with dataset naming convention of HDF5 files from erikbern/ann-benchmarks.
    Condition to detect groundtruth dataset needs to be adjusted when used with HDF5 files from elsewhere.
"""

def hdf5_to_xvecs(hdf5_file):
    # Open the HDF5 file
    datasets = []
    with h5py.File(hdf5_file, "r") as f:
        # Get dataset names
        f.visititems(lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None)
        for dataset in datasets:
            # Read the specified dataset
            data = f[dataset][:]
            
            # Check if this is groundtruth data (should be integers)
            if dataset.lower() == "neighbors":
                # Convert to int32 for ivecs
                data = data.astype(np.int32)
                file_extension = ".ivecs"
            else:
                # Convert to float32 for fvecs
                data = data.astype(np.float32)
                file_extension = ".fvecs"
            
            # Write to fvecs/ivecs file
            output_file = os.path.split(os.path.abspath(hdf5_file))[0] + "/" + dataset + file_extension
            
            with open(output_file, "wb") as outfile:
                for vector in data:
                    # Write the dimensionality (as int32)
                    dimension = np.int32(len(vector))
                    dimension.tofile(outfile)
                    # Write the vector data
                    vector.tofile(outfile)

def main():
    if len(sys.argv) < 2:
        print("Usage: python hdf5_to_xvecs.py <hdf5_filename>.\nFor better file organization, make sure the HDF5 file is in its own folder")
        return
    hdf5_to_xvecs(sys.argv[1])

if __name__ == '__main__':
    main()