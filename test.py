import torch

# res = torch.cuda.is_available()
# print(res)

# res = torch.backends.mps.is_available()
# print(res)

x = torch.tensor(7)
print(x)

y = x.ndim
print(y)

z = x.item()
print(z)

vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim)