# Next step will be to implement the self-attention mechanism used in the original transformer 
# architecture, the GPT models, and most other popular LLMs. This self-attention mechanism 
# is also called scaled dot-product attention.
import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)    
     [0.55, 0.87, 0.66], # journey  (x^2)    
     [0.57, 0.85, 0.64], # starts   (x^3)    
     [0.22, 0.58, 0.33], # with     (x^4)    
     [0.77, 0.25, 0.10], # one      (x^5)    
     [0.05, 0.80, 0.55]] # step     (x^6) 
    )

query = inputs[1] #1 the second input element
d_in = inputs.shape[1] #2 the input embedding size
d_out = 2 #3 the output embedding size

print(f"inputs.shape[1]: {d_in}")
torch.manual_seed(123) 
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) 
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) 
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print(f"Query: {W_query}. Key: {W_key}. Value: {W_value}")


query_2 = query @ W_query 
key_2 = query @ W_key 
value_2 = query @ W_value 

print(f"Query: {query_2}. Key: {key_2}. Value: {value_2}")


keys = inputs @ W_key 
values = inputs @ W_value

print(f"Keys: {keys}. Values: {values}")

# Attention score
keys_2 = keys[1] #1 
attn_score_22 = query_2.dot(keys_2) 

print(f"Single attention score: {attn_score_22}")

attn_scores_2 = query_2 @ keys.T #1 
print(f"Full attention score: {attn_scores_2}")


# Attention weights

d_k = keys.shape[-1] 
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) 
print(attn_weights_2)

# Context vector
context_vec_2 = attn_weights_2 @ values 
print(f"Context vector: {context_vec_2}")