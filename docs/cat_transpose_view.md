`torch.cat` is a **PyTorch function** used to **concatenate tensors** along a given dimension.

In simple terms: it takes a sequence of tensors and joins them together into a single tensor.

---

### Syntax:

```python
torch.cat(tensors, dim=0, *, out=None)
```

### Parameters:

- **tensors**: A sequence (list/tuple) of tensors that you want to concatenate.

  - They must all have the same shape _except_ in the dimension you're concatenating along.

- **dim**: The dimension (axis) along which to concatenate.
- **out** _(optional)_: An output tensor to store the result.

---

### Example 1: Concatenating along rows

```python
import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6]])
result = torch.cat((a, b), dim=0)

print(result)
```

**Output:**

```
tensor([[1, 2],
        [3, 4],
        [5, 6]])
```

Here, we concatenated along `dim=0` (rows).

---

### Example 2: Concatenating along columns

```python
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
result = torch.cat((a, b), dim=1)

print(result)
```

**Output:**

```
tensor([[1, 2, 5, 6],
        [3, 4, 7, 8]])
```

Here, we concatenated along `dim=1` (columns).

---

✅ **Key Difference**:

- `torch.cat` joins _existing dimensions_.
- If you want to add a _new dimension_ instead, you use **`torch.stack`**.

---

Do you want me to also explain **`torch.cat` vs `torch.stack`** with examples (since people often confuse them)?

Great question! In PyTorch, **`torch.transpose`** swaps two dimensions of a tensor.

---

### Syntax

```python
torch.transpose(input, dim0, dim1) → Tensor
```

- **`input`** → the tensor you want to transpose.
- **`dim0`** → the first dimension to swap.
- **`dim1`** → the second dimension to swap.

It returns a **view** of the original tensor (no data is copied).

---

### Example 1: 2D Tensor (like a matrix)

```python
import torch

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

y = torch.transpose(x, 0, 1)  # swap rows and columns
print(y)
```

**Output:**

```
tensor([[1, 4],
        [2, 5],
        [3, 6]])
```

This is the same as taking the matrix transpose.

---

### Example 2: Higher-dimensional tensor

```python
x = torch.randn(2, 3, 4)  # shape (2, 3, 4)

y = torch.transpose(x, 1, 2)  # swap dim 1 and dim 2
print(x.shape)  # torch.Size([2, 3, 4])
print(y.shape)  # torch.Size([2, 4, 3])
```

Here, the last two dimensions are swapped.

---

### ⚡ Shortcut

For **2D tensors only**, you can use `.t()`:

```python
x.t()  # equivalent to torch.transpose(x, 0, 1)
```

---

✅ **Key point**:

- `torch.transpose` just swaps two dimensions.
- If you want to reorder **multiple dimensions**, you use **`torch.permute`**.

---

Do you want me to show you a quick **transpose vs permute** comparison? That’s where people usually get tripped up.

In PyTorch, **`view`** is used to **reshape a tensor** without changing its underlying data.

Think of it as telling PyTorch: _"I want to look at the same data in a different shape."_

---

### Syntax

```python
tensor.view(*shape) → Tensor
```

- **`*shape`**: the new shape you want.
- The new shape must have the **same number of elements** as the original tensor.

---

### Example 1: Reshaping a 1D tensor

```python
import torch

x = torch.arange(6)        # tensor([0, 1, 2, 3, 4, 5])
y = x.view(2, 3)           # reshape to 2x3
print(y)
```

**Output:**

```
tensor([[0, 1, 2],
        [3, 4, 5]])
```

---

### Example 2: Using `-1` (automatic inference)

You can use `-1` to let PyTorch figure out the dimension size automatically.

```python
x = torch.arange(12)       # tensor with 12 elements
y = x.view(3, -1)          # -1 tells PyTorch to infer the size
print(y.shape)             # torch.Size([3, 4])
```

---

### Example 3: Higher dimensions

```python
x = torch.randn(2, 3, 4)   # shape (2, 3, 4) → total 24 elements
y = x.view(4, 6)           # reshape to (4, 6)
print(y.shape)             # torch.Size([4, 6])
```

---

### ⚡ Key Notes

- `view` **does not copy** the data → it gives a new view of the same memory (fast!).
- But it only works when the tensor is **contiguous in memory**.

  - If it’s not contiguous (e.g., after a `transpose`), you may need `.contiguous()` before `.view()`.
  - Example: `x.transpose(0,1).contiguous().view(...)`

---

✅ In short:

- **`view`** = reshape (same data, different shape).
- **`reshape`** (another function) is similar but more flexible—it can handle non-contiguous tensors by making a copy if needed.

---

Would you like me to also break down the **difference between `view` and `reshape`** with examples? That’s something that often confuses people.
