import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)    
     [0.55, 0.87, 0.66], # journey  (x^2)    
     [0.57, 0.85, 0.64], # starts   (x^3)    
     [0.22, 0.58, 0.33], # with     (x^4)    
     [0.77, 0.25, 0.10], # one      (x^5)    
     [0.05, 0.80, 0.55]] # step     (x^6) 
    )

query = inputs[1] #1 journey

# The first step of implementing self-attention is to compute the intermediate values w, referred to as attention scores,
attention_scores = torch.empty(inputs.shape[0])
for input_index, input_row in enumerate(inputs):
    attention_scores[input_index] = torch.dot(input_row, query) # torch([0.55, 0.87, 0.66]) * torch([0.43, 0.15, 0.89])

print(f"Attention scores after dot product: {attention_scores}")

# In the next step, as shown in figure 3.9, we normalize each of the attention scores we computed previously. 
# The main goal behind the normalization is to obtain attention weights that sum up to 1. 
# This normalization is a convention that is useful for interpretation and maintaining training stability in an LLM. 

# 
attn_weights = torch.softmax(attention_scores, dim=0) 
# 
# attn_weights = attention_scores / attention_scores.sum() # tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
print(f"Attention weights: {attn_weights}")
# print("Sum:", attn_weights.sum()) # tensor(1.0000)

# The final step, after calculating and normalizing the attention scores to obtain the attention weights for query x(2), 
# is to compute the context vector z(2). This context vector is a combination of all input vectors x(1) to x(T ) weighted by the attention weights.
context_vector = torch.zeros(query.shape)
for input_index,input_row in enumerate(inputs):
    context_vector += attn_weights[input_index]*input_row # tensor(0.1455) * tensor([0.43, 0.15, 0.89])
    
print("Context Vector:", context_vector)


# ********* Same as: (inputs @ inputs.T)
# attn_scores = torch.empty(6, 6) 
# for i, x_i in enumerate(inputs): 
#     for j, x_j in enumerate(inputs): 
#         attn_scores[i, j] = torch.dot(x_i, x_j) 
attn_scores = inputs @ inputs.T
print(f"Attention scores: {attn_scores}")
attn_weights = torch.softmax(attn_scores, dim=1)
print(f"Attention weights: {attn_weights}")
all_context_vecs = attn_weights @ inputs
print(f"All Context vectors: {all_context_vecs}")