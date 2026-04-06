from gelu import GELU

import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
#A
out = ffn(x)
print(out.shape)


# The FeedForward module we implemented in this section plays a crucial role in enhancing
# the model's ability to learn from and generalize the data. Although the input and output
# dimensions of this module are the same, it internally expands the embedding dimension
# into a higher-dimensional space through the first linear layer as illustrated in Figure 4.10.
# This expansion is followed by a non-linear GELU activation, and then a contraction back to
# the original dimension with the second linear transformation. Such a design allows for the
# exploration of a richer representation space.