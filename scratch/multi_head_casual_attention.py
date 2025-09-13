
import torch
import torch.nn as nn
from data_inputs import inputs

# Input dimension is derived from the shape of the input tensor (embedding size per token)
input_embedding_dimension = inputs.shape[1]  # Example: 2
# Output dimension is the dimensionality of the transformed embedding space
output_embedding_dimension = 2  # Example: 3

class MultiHeadAttention(nn.Module):    
    """
    Implements multi-head scaled dot-product self-attention with a causal mask.

    Args:
        input_dimensions (int): Dimensionality of input embeddings.
        output_dimensions (int): Dimensionality of output embeddings.
                                 Must be divisible by num_heads.
        context_length (int): Maximum sequence length (used for causal mask).
        dropout (float): Dropout probability applied to attention weights.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, includes bias in Q, K, V projections. Default: False.

    Attributes:
        W_query (nn.Linear): Linear projection for queries.
        W_key (nn.Linear): Linear projection for keys.
        W_value (nn.Linear): Linear projection for values.
        out_proj (nn.Linear): Linear layer to project concatenated head outputs.
        dropout (nn.Dropout): Dropout applied to attention weights.
        mask (Tensor): Upper-triangular mask for causal attention
                       (prevents attending to future tokens).
    """
        
    def __init__(self, input_dimensions, output_dimensions, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # assert output_dimensions % num_heads == 0, "output_dimensions must be divisible by num_heads"

        self.output_dimensions = output_dimensions
        self.num_heads = num_heads
        self.head_dim = output_dimensions // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(input_dimensions, output_dimensions, bias=qkv_bias)
        self.W_key = nn.Linear(input_dimensions, output_dimensions, bias=qkv_bias)
        self.W_value = nn.Linear(input_dimensions, output_dimensions, bias=qkv_bias)
        self.out_proj = nn.Linear(output_dimensions, output_dimensions)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, input_dimensions = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, output_dimensions)
        queries = self.W_query(x)
        values = self.W_value(x)

        # print("keys", keys)
        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, output_dimensions) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # print("keys.view", keys)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        # print("keys.transpose", keys)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head
        
        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.output_dimensions = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.output_dimensions)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec
    
    

# Parameters
torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]
dropout=0.0
num_heads=2

# Initialize attention layer
mha = MultiHeadAttention(
    input_embedding_dimension,
    output_embedding_dimension,
    context_length,
    dropout,
    num_heads
)

# Forward pass
context_vector = mha.forward(batch)

print("Context vector", context_vector)
print("Context vector shape:", context_vector.shape)

# print("Input shape:", x.shape)
# print("Output shape:", output)