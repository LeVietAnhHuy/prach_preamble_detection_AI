# import numpy as np
# import os
# import tqdm
#
# corr_data_path = "generated_dataset/corr_dataset"
# # List of files in desired order
# file_list = [
#     "corr_0dB.npy",
#     "corr_5dB.npy",
#     "corr_10dB.npy",
#     "corr_15dB.npy",
#     "corr_-5dB.npy",
#     "corr_-10dB.npy",
#     "corr_-15dB.npy",
#     "corr_-20dB.npy",
#     "corr_-25dB.npy",
#     "corr_-30dB.npy",
# ]
#
# # Load and concatenate
# arrays = [np.load(os.path.join(corr_data_path, f)) for f in file_list]
# merged = np.concatenate(arrays, axis=0)
#
# # Save the merged array
# np.save(os.path.join(corr_data_path, "merged_corr.npy"), merged)
#
# print("Merged shape:", merged.shape)

import numpy as np
import os

# Directory containing the .npy files
corr_data_path = "generated_dataset/corr_dataset"

# Ordered list of filenames (adjust if needed)
file_list = [
    "corr_0dB.npy",
    "corr_5dB.npy",
    "corr_10dB.npy",
    "corr_15dB.npy",
    "corr_-5dB.npy",
    "corr_-10dB.npy",
    "corr_-15dB.npy",
    "corr_-20dB.npy",
    "corr_-25dB.npy",
    "corr_-30dB.npy",
]

# Full paths
file_paths = [os.path.join(corr_data_path, f) for f in file_list]

# Load shape and dtype from the first file (mmap_mode avoids full load)
sample = np.load(file_paths[0], mmap_mode='r')
x, y = sample.shape
dtype = sample.dtype
n_files = len(file_paths)

# Create memmap array to store the merged data
merged_shape = (n_files * x, y)
merged = np.memmap(os.path.join(corr_data_path, "merged_corr.npy"), dtype=dtype, mode="w+", shape=merged_shape)

# Write each file into the correct block of the memmap
for i, file in enumerate(file_paths):
    data = np.load(file, mmap_mode="r")
    merged[i * x : (i + 1) * x, :] = data
    del data  # optional: clean up memory

# Flush to ensure it's written to disk
merged.flush()

print("Merged array saved as 'merged_corr.npy'. Shape:", merged.shape)
