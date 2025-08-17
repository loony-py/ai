import torch
# import pandas
# import numpy
# import matplotlib.pyplot

print(torch.__version__)

# Tensors

# https://docs.pytorch.org/docs/stable/tensors.html

# * Scalar
scalar = torch.tensor(7)
print(f"Scalar tensor value: {scalar.item()}")
print(f"Scalar dimensions: {scalar.ndim}")
print("\n")

# * Vector
vector = torch.tensor([0, 0])
print(f"Vector: {vector}")
print(f"Vector dimensions: {vector.ndim}")
print(f"Vector shape: {vector.shape}")
print(f"Vector size: {vector.size()}")
print("\n")

# * Matrix

matrix = torch.tensor([
    [1, 2],
    [3, 4]
])
print(f"Matrix: {matrix}")
print(f"Matrix dimensions: {matrix.ndim}")
print(f"Matrix index 0: {matrix[0]}")
print(f"Matrix shape: {matrix.shape}")
print(f"Matrix size: {matrix.size()}")
print("\n")

# * Tensors
TENSOR = torch.tensor([
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ],
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ],
])
print(f"Tensor: {TENSOR}")
print(f"Tensor dimensions: {TENSOR.ndim}")
print(f"Tensor shape: {TENSOR.shape}")
print(f"Tensor size: {TENSOR.size()}")
print(f"Tensor index 0: {TENSOR[1]}")