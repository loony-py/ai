import torch

# * Random tensors

random_tensor = torch.rand(size=(224, 224, 3))
print(f"Random tensor shape: {random_tensor.shape}")
print(f"Random tensor dimensions: {random_tensor.dim()}")
print("\n")

# * Zeroes and Ones

zeroes = torch.zeros(3, 3)
print(f"Zeroes: {zeroes}")
print(f"Zeroes tensor shape: {zeroes.shape}")
print(f"Zeroes tensor dimensions: {zeroes.dim()}")
print("\n")

ones = torch.ones(3, 3)
print(f"Ones: {ones}")
print(f"Ones tensor shape: {ones.shape}")
print(f"Ones tensor dimensions: {ones.dim()}")
print(f"Datatype: {ones.dtype}")
print("\n")

# Range tensor
zero_to_9 = torch.arange(0, 10)
one_to_ten_with_step = torch.arange(start=1, end=11, step=2)
print(f"Range tensor: {zero_to_9}")
print(f"Range tensor dimensions: {zero_to_9.dim()}")
print(f"Range tensor with step: {one_to_ten_with_step}")
print("\n")

# Creating tensors like

ten_zeroes = torch.zeros_like(input=zero_to_9)
print(f"Zeroes like: {ten_zeroes}")