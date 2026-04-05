import torch 
import torch.nn as nn 
import tiktoken

torch.manual_seed(123)
batch_example = torch.randn(2, 5) #A

#A create 2 training examples with 5 dimensions (features) each

# ReLU Rectified Linear Unit
# it simply thresholds negative
# inputs to 0
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU()) 
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)


# Using keepdim=True in operations like mean or variance calculation ensures that the
# output tensor retains the same number of dimensions as the input tensor, even though the
# operation reduces the tensor along the dimension specified via dim. For instance, without
# keepdim=True, the returned mean tensor would be a 2-dimensional vector [0.1324,
# 0.2170] instead of a 2×1-dimensional matrix [[0.1324], [0.2170]].

# The dim parameter specifies the dimension along which the calculation of the statistic
# (here, mean or variance) should be performed in a tensor.
# dim=1 or dim=-1 calculates mean across the column dimension to obtain one mean per row
# dim=0 calculates mean across the row dimension to obtain one mean per row

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
torch.set_printoptions(sci_mode=False)

print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        #   The variable eps is a small constant (epsilon) 
        # added to the variance to prevent division by zero during
        # normalization.
        self.eps = 1e-5
        #   The scale and shift are two trainable parameters (of the same dimension
        # as the input) that the LLM automatically adjusts during training if it is determined that
        # doing so would improve the model's performance on its training task. This allows the model
        # to learn appropriate scaling and shifting that best suit the data it is processing.
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    # This specific implementation of layer Normalization operates on the last dimension of the
    # input tensor x, which represents the embedding dimension (emb_dim).
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)