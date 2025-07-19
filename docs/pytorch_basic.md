Here’s a **comprehensive list of basic PyTorch commands**, organized by category for easier reference. This covers the fundamentals you'll use frequently when working with PyTorch.

---

### 🔧 1. **Installation**

```bash
pip install torch torchvision torchaudio
```

---

### 🔢 2. **Tensors**

#### Create Tensors

```python
import torch

# From data
x = torch.tensor([1.0, 2.0, 3.0])
# Zeros, ones, random
torch.zeros(2, 3)
torch.ones(2, 3)
torch.rand(2, 3)
torch.randn(2, 3)
# Identity and arange
torch.eye(3)
torch.arange(0, 10, 2)
```

#### Tensor Properties

```python
x.shape
x.dtype
x.device
```

#### Data Types

```python
x.float()
x.int()
x.long()
```

#### Reshaping

```python
x.view(3, 1)
x.reshape(3, 1)
x.squeeze()   # remove dimensions of size 1
x.unsqueeze(0)  # add dimension
```

#### Indexing/Slicing

```python
x[0]
x[:2]
x[:, 1]
```

#### Tensor Math

```python
a + b
a - b
a * b
a / b
a @ b.T   # matrix multiplication
torch.matmul(a, b)
torch.dot(a, b)
```

---

### 🎛️ 3. **GPU Support**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
```

---

### 🔄 4. **Autograd (Automatic Differentiation)**

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
x.grad  # dy/dx
```

---

### 🏗️ 5. **Neural Network Building**

#### Define a Model

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = Net()
```

#### Loss & Optimizer

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

#### Training Loop Skeleton

```python
for epoch in range(epochs):
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
```

---

### 📦 6. **Data Loading**

#### Dataset & DataLoader

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(x_data, y_data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

#### Built-in Datasets

```python
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
```

---

### 💾 7. **Saving and Loading Models**

#### Save

```python
torch.save(model.state_dict(), 'model.pth')
```

#### Load

```python
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

---

### 🧪 8. **Evaluation**

```python
model.eval()
with torch.no_grad():
    output = model(x)
```

---

### 📊 9. **Tensor Utilities**

```python
x.item()           # get Python number from a single-element tensor
x.numpy()          # to NumPy
torch.from_numpy(numpy_array)  # from NumPy
torch.cat((a, b), dim=0)       # concatenate
torch.stack([a, b], dim=0)     # stack
```

---

Would you like a **printable cheat sheet** PDF version of this as well?
