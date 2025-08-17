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
