import torch

# res = torch.cuda.is_available()
# print(res)

# res = torch.backends.mps.is_available()
# print(res)

a = torch.tensor([[[1, 2, 3, 4]]])
b = torch.tensor([
    [4, 5, 6],
    [4, 5, 6],
    [4, 5, 6],
    [4, 5, 6],
])
d = torch.matmul(a, b)
print(d)

# a = torch.rand(3, 5)
# b = torch.rand(3, 5)

# print(torch.matmul(a, b))
# print(a)
# print(b)