Got it — you want a **PyTorch command toolkit** for all the essential math and tensor operations you’ll actually use in ML/AI.

Here’s a **beginner-friendly but complete** reference so you can keep it handy while coding.

---

## **1. Tensor Creation**

```python
import torch

# From Python lists
a = torch.tensor([1, 2, 3], dtype=torch.float32)

# Zeros & ones
torch.zeros(3, 3)   # 3x3 zero matrix
torch.ones(2, 4)    # 2x4 ones

# Random
torch.rand(2, 3)    # uniform random [0,1)
torch.randn(2, 3)   # normal distribution

# Identity matrix
torch.eye(3)        # 3x3 identity

# Range
torch.arange(0, 10, 2)  # 0, 2, 4, 6, 8
torch.linspace(0, 1, 5) # evenly spaced numbers
```

---

## **2. Reshaping & Transposing**

```python
x = torch.arange(1, 7).reshape(2, 3)  # reshape
x.T                                    # transpose
x.view(-1)                             # flatten
x.unsqueeze(0)                         # add dimension
x.squeeze()                            # remove dimension
```

---

## **3. Basic Math**

```python
a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)

# Elementwise operations
a + b
a - b
a * b
a / b
a ** 2         # power

# In-place ops (modifies a directly)
a.add_(b)
a.mul_(b)

# Common functions
torch.sqrt(a)
torch.exp(a)
torch.log(a)
torch.abs(a)
torch.sin(a)
torch.cos(a)
```

---

## **4. Matrix Operations**

```python
A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

# Matrix multiplication
torch.matmul(A, B)   # or A @ B

# Elementwise multiplication
A * B

# Dot product (1D vectors)
u = torch.tensor([1, 2, 3], dtype=torch.float32)
v = torch.tensor([4, 5, 6], dtype=torch.float32)
torch.dot(u, v)   # 1*4 + 2*5 + 3*6

# Transpose
A.T

# Inverse
torch.inverse(A)

# Determinant
torch.det(A)
```

---

## **5. Statistics & Aggregations**

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

x.sum()            # total sum
x.mean()           # average
x.std()            # standard deviation
x.min(), x.max()   # min & max
x.argmin(), x.argmax()  # indices of min & max

# Along a dimension
x.sum(dim=0)  # column-wise sum
x.sum(dim=1)  # row-wise sum
```

---

## **6. Comparison & Logical Ops**

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([3, 2, 1])

a == b
a != b
a > b
a < b

torch.all(a < b)    # all elements satisfy condition
torch.any(a < b)    # any element satisfies condition
```

---

## **7. GPU Operations**

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.rand(3, 3).to(device)
y = torch.rand(3, 3).to(device)
z = x + y
```

---

## **8. Useful Extras**

```python
torch.clamp(x, min=0)       # clip values below 0
torch.where(a > b, a, b)    # elementwise conditional
torch.topk(a, k=2)          # top 2 values
torch.sort(a)               # sort values
```

---

✅ This set of commands covers **90%** of the PyTorch tensor work you’ll do in ML/AI — from data preprocessing to forward passes in neural networks.

If you want, I can also make you a **visual PyTorch cheatsheet** with all these commands laid out like a quick-reference poster — perfect for beginners. That way you don’t have to scroll through code blocks.
