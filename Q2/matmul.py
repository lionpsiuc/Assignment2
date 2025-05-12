import struct
import numpy as np

matrix_file = "mat-d20-b5-p4.bin"
vector_file = "x-d20.txt.bin"
block_dim = 5

# Read matrix data
with open(matrix_file, "rb") as f:

    # Read dimension
    dim_bytes = f.read(4)
    dim = struct.unpack("i", dim_bytes)[0]

    # Calculate number of blocks per dimension
    num_blocks = dim // block_dim

    # Initialize an empty matrix
    matrix = np.empty((dim, dim), dtype=np.float64)

    # Read matrix blocks based on block-column storage
    for col_block_idx in range(num_blocks):
        for row_block_idx in range(num_blocks):

            # Calculate the starting position for the current block in the file
            offset = 4 + (col_block_idx * num_blocks * block_dim * block_dim * 8) + (row_block_idx * block_dim * block_dim * 8)
            f.seek(offset)

            # Read the current block data
            block_data = f.read(block_dim * block_dim * 8)
            block = np.frombuffer(block_data, dtype=np.float64).reshape((block_dim, block_dim))

            # Place the block in the correct position in the matrix
            row_start = row_block_idx * block_dim
            row_end = row_start + block_dim
            col_start = col_block_idx * block_dim
            col_end = col_start + block_dim
            matrix[row_start:row_end, col_start:col_end] = block

# Read vector data
with open(vector_file, "rb") as f:
    length_bytes = f.read(4)
    length = struct.unpack("i", length_bytes)[0]

    vector_data = f.read(length * 8) # 8 bytes for float64
    vector = np.frombuffer(vector_data, dtype=np.float64)

result = np.dot(matrix, vector)

print(result)
