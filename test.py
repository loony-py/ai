import torch

# From Python lists
a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.zeros(3, 3)   # 3x3 zero matrix
c = torch.ones(3, 4)    # 2x4 ones

print("Tensor a:", a)
print("Tensor b:", b)
print("Tensor c", c)
print("Tensor c", c.size(), c.dim())
