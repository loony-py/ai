import torch
import torch.nn as nn 
from data_inputs import inputs

input_dimension = inputs.shape[1] #2 the input embedding size
output_dimension = 2 #3 the output embedding size

class SelfAttention_V1(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__() 
        self.W_query = nn.Parameter(torch.rand(input_dimension, output_dimension)) 
        self.W_key = nn.Parameter(torch.rand(input_dimension, output_dimension))
        self.W_value = nn.Parameter(torch.rand(input_dimension, output_dimension))
        
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(123)
self_attention = SelfAttention_V1(input_dimension, output_dimension)
res = self_attention.forward(inputs)
print(f"Context vector: {res}")


class SelfAttention_2(nn.Module):
    def __init__(self, input_dimension, output_dimension, qkv_bias=False):
        super().__init__() 
        # self.W_query = nn.Parameter(torch.rand(input_dimension, output_dimension)) 
        # self.W_key = nn.Parameter(torch.rand(input_dimension, output_dimension))
        # self.W_value = nn.Parameter(torch.rand(input_dimension, output_dimension))
        self.W_query = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.W_key   = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.W_value = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(789)
self_attention = SelfAttention_2(input_dimension, output_dimension)
res = self_attention.forward(inputs)
print(f"Context vector: {res}")

queries = self_attention.W_query(inputs)
keys = self_attention.W_key(inputs)
attn_scores = queries @ keys.T 
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) 

context_length = attn_scores.shape[0] 
mask_simple = torch.tril(torch.ones(context_length, context_length)) 
masked_simple = attn_weights*mask_simple 
row_sums = masked_simple.sum(dim=-1, keepdim=True) 
masked_simple_norm = masked_simple / row_sums 

print(f"\nQueries: {queries}")
print(f"\nKeys: {keys}")
print(f"\nAttention Weights: {attn_weights}")
print(f"\nMasked simple: {masked_simple}")
print(f"\n{masked_simple_norm}")
