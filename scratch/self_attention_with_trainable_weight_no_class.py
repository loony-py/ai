# ------------------------------
# Self-Attention Mechanism (Step-by-step)
# ------------------------------

import torch

# ------------------------------
# 1. Input embeddings (toy example sentence)
# ------------------------------
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # "Your"   (x^1)
    [0.55, 0.87, 0.66],  # "journey" (x^2)
    [0.57, 0.85, 0.64],  # "starts"  (x^3)
    [0.22, 0.58, 0.33],  # "with"    (x^4)
    [0.77, 0.25, 0.10],  # "one"     (x^5)
    [0.05, 0.80, 0.55]   # "step"    (x^6)
])

print("Input embeddings shape:", inputs.shape)  # (6 words × 3 embedding dims)

# ------------------------------
# 2. Select a query word
# ------------------------------
context_word = inputs[1]   # "journey"
print("\nQuery word embedding (\"journey\"):", context_word)

# ------------------------------
# 3. Define projection dimensions
# ------------------------------
input_dimension = inputs.shape[1]   # input embedding size = 3
output_dimension = 2                # output projection size = 2

# ------------------------------
# 4. Initialize weight matrices
# ------------------------------
torch.manual_seed(123)   # reproducibility
random_query_weight = torch.rand(input_dimension, output_dimension)   # Query projection
random_key_weight = torch.rand(input_dimension, output_dimension)   # Key projection
random_value_weight = torch.rand(input_dimension, output_dimension)   # Value projection

print("\nRandom Weights\n:")
print("random_query_weight:", random_query_weight)
print("random_key_weight:", random_key_weight)
print("random_value_weight:", random_value_weight)

# ------------------------------
# 5. Compute Q, K, V for the query word
# ------------------------------
queryweight_for_context = context_word @ random_query_weight
keyweight_for_context = context_word @ random_key_weight
valueweight_for_context = context_word @ random_value_weight

print("\nProjected vectors for \"journey\":")
print("Query Weight for query (Q):", queryweight_for_context)
print("Key Weight for query (K):", keyweight_for_context)
print("Value Weight for query (V):", valueweight_for_context)

# ------------------------------
# 6. Compute K and V for all words
# ------------------------------
all_key_weights = inputs @ random_key_weight    # shape (6 × 2)
all_value_weights = inputs @ random_value_weight    # shape (6 × 2)

print("\nAll Keys:\n", all_key_weights)
print("All Values:\n", all_value_weights)

# ------------------------------
# 7. Attention scores (dot product)
# ------------------------------
# Compare query of "journey" with all keys
attention_scores = queryweight_for_context @ all_key_weights.T   # shape (6,)
print("\nAttention scores (before scaling):", attention_scores)

# ------------------------------
# 8. Scale and normalize with softmax
# ------------------------------
all_key_weights_dimension = all_key_weights.shape[-1]   # key dimension (2)
scaled_scores = attention_scores / (all_key_weights_dimension ** 0.5)
attention_weights = torch.softmax(scaled_scores, dim=-1)

print("\nScaled attention scores:", scaled_scores)
print("Attention weights (softmax):", attention_weights)

# ------------------------------
# 9. Compute context vector
# ------------------------------
context_vector = attention_weights @ all_value_weights
print("\nFinal context vector for \"journey\":", context_vector)



###################################################################################################
# Input embeddings shape: torch.Size([6, 3])

# Query word embedding ("journey"): tensor([0.5500, 0.8700, 0.6600])

# Projected vectors for "journey":
# Query Weight for query (Q): tensor([0.4306, 1.4551])
# Key Weight for query (K): tensor([0.4433, 1.1419])
# Value Weight for query (V): tensor([0.3951, 1.0037])

# All Keys:
#  tensor([[0.3669, 0.7646],
#         [0.4433, 1.1419],
#         [0.4361, 1.1156],
#         [0.2408, 0.6706],
#         [0.1827, 0.3292],
#         [0.3275, 0.9642]])
# All Values:
#  tensor([[0.1855, 0.8812],
#         [0.3951, 1.0037],
#         [0.3879, 0.9831],
#         [0.2393, 0.5493],
#         [0.1492, 0.3346],
#         [0.3221, 0.7863]])

# Attention scores (before scaling): tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])

# Scaled attention scores: tensor([0.8984, 1.3098, 1.2806, 0.7633, 0.3944, 1.0918])
# Attention weights (softmax): tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])

# Final context vector for "journey": tensor([0.3061, 0.8210])