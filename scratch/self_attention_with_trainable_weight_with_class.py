import torch
import torch.nn as nn 
from data_inputs import inputs

query = inputs[1] #1 the second input element
d_in = inputs.shape[1] #2 the input embedding size
d_out = 2 #3 the output embedding size

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__() 
        self.W_query = nn.Parameter(torch.rand(d_in, d_out)) 
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        
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
sa_v1 = SelfAttention_v1(d_in, d_out)
res = sa_v1.forward(inputs)
print(f"Context vector: {res}")