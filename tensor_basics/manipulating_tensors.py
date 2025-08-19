import torch

# Manipulating tensors (tensor operations)
# * Addition
# * Substraction
# * Multiplication (element-wise)
# * Division
# * Matrix multiplication

# Addition
tensor_ = torch.tensor([1, 2, 3])
add_10 = tensor_ + 10
mul_10 = tensor_ * 10
sub_10 = tensor_ - 10

print(f"Add tensor: {add_10}")
print(f"Mul tensor: {mul_10}")
print(f"Sub tensor: {sub_10}")
print("\n")

# In built functions

add_10 = torch.add(tensor_, 10)
print(f"Add tensor using inbuilt-fn: {add_10}")
mul_10 = torch.mul(tensor_, 10)
print(f"Mul tensor using inbuilt-fn: {mul_10}")
div_5 = torch.div(torch.tensor([10, 20, 30]), 5)
print(f"Div tensor using inbuilt-fn: {div_5}")
print("\n")

## Links
# https://www.mathsisfun.com/algebra/matrix-multiplying.html


# Matrix multiplication

mat_one = torch.tensor([1, 2, 3])
mat_two = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
mat_mul = torch.matmul(mat_one, mat_two)
print(f"Matrix multiplication: {mat_mul}")