# 🧠 Tutorial: Building the Attention Mechanism from Scratch

This notebook walks through a **step-by-step implementation of the self-attention mechanism**, the core component of Transformer-based Large Language Models (LLMs).

We’ll take a toy sentence, represent each word with small embeddings, and manually compute Queries, Keys, Values, attention scores, and the resulting context vector.

---

## 1. Setup

```python
import torch
```

We’ll be working with PyTorch tensors to represent embeddings and perform matrix multiplications.

---

## 2. Input Embeddings

```python
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your   (x^1)
     [0.55, 0.87, 0.66], # journey (x^2)
     [0.57, 0.85, 0.64], # starts  (x^3)
     [0.22, 0.58, 0.33], # with    (x^4)
     [0.77, 0.25, 0.10], # one     (x^5)
     [0.05, 0.80, 0.55]] # step    (x^6)
)
```

- Each row is a **word embedding** (dimension = 3).
- There are **6 words** in the sentence.
- Shape of `inputs`: **(6 × 3)**.

📌 **Goal**: Build self-attention for the word **"journey"**.

---

## 3. Select the Query Word

```python
query = inputs[1]  # "journey"
```

We will compute how much "journey" should pay attention to each other word in the sentence.

---

## 4. Define Dimensions and Weight Matrices

```python
d_in = inputs.shape[1]  # input embedding size = 3
d_out = 2               # projected dimension = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
```

- We project embeddings (3D → 2D) using **learnable matrices**.
- These matrices usually train with gradient descent, but here we freeze them for demonstration.

---

## 5. Compute Query, Key, Value Vectors

```python
query_2 = query @ W_query
key_2   = query @ W_key
value_2 = query @ W_value

print("Query:", query_2)
print("Key:", key_2)
print("Value:", value_2)
```

- **Query (Q)** → What information the word is looking for.
- **Key (K)** → What information the word offers.
- **Value (V)** → The actual representation to share.

Shape after projection: **(2,)**.

---

## 6. Compute Keys and Values for All Words

```python
keys   = inputs @ W_key
values = inputs @ W_value

print("Keys:", keys)
print("Values:", values)
```

- Keys: shape **(6 × 2)**
- Values: shape **(6 × 2)**

Now every word has a **K** and **V**.

---

## 7. Attention Score (Single Pair)

```python
keys_2 = keys[1]  # key for "journey"
attn_score_22 = query_2.dot(keys_2)
print("Attention score (journey vs journey):", attn_score_22)
```

This dot product measures similarity between the query of _"journey"_ and the key of _"journey"_.

---

## 8. Attention Scores with All Words

```python
attn_scores_2 = query_2 @ keys.T
print("All attention scores:", attn_scores_2)
```

- Compare **"journey"** (query) with **all keys**.
- Output shape: **(6,)** → one score per word.

---

## 9. Scale and Normalize with Softmax

```python
d_k = keys.shape[-1]  # dimensionality of keys = 2
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("Attention weights:", attn_weights_2)
```

- Scale by √d_k to prevent large values.
- Softmax → probabilities (weights sum to 1).

This tells us: _how much attention "journey" gives to each word_.

---

## 10. Compute Context Vector

```python
context_vec_2 = attn_weights_2 @ values
print("Context vector for 'journey':", context_vec_2)
```

- Weighted sum of **values**, using attention weights.
- Shape: **(2,)**.
- This is the **contextual embedding** of "journey", enriched by surrounding words.

---

## 🔎 Big Picture

1. Each word → Query, Key, Value.
2. Queries compare with Keys → scores.
3. Scores → normalized weights.
4. Weights mix Values → context vector.

👉 This allows words to **dynamically focus** on others depending on context.
👉 In full Transformers, this is done **for every word in parallel** and extended with **multi-head attention**.

---

## 🎯 Why This Matters

Without attention, models treat context rigidly (like fixed windows or sequential recurrence).
With attention, each word can "look around" and decide what matters most.

Example: In _"Your journey starts with one step"_, the model can:

- Link **"journey"** with **"starts"** (semantic connection).
- Ignore less relevant words.

This mechanism is why LLMs scale so effectively.

---

✅ **Summary**: We manually built **the self-attention mechanism for one word ("journey")**. The output `context_vec_2` is the context-aware representation of that word.

---
