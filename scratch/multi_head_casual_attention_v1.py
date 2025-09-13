
import torch
import torch.nn as nn
from data_inputs import inputs



import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    """
    Implements **causal self-attention** (a masked scaled dot-product attention) 
    for sequence modeling tasks such as language modeling.

    This attention mechanism ensures that each position in the sequence
    can only attend to *past* or *current* tokens, never future ones.
    This is done using a causal mask (upper-triangular mask).

    Parameters
    ----------
    input_dimension : int
        Dimensionality of the input embeddings.
    output_dimension : int
        Dimensionality of the projected queries, keys, and values.
    context_length : int
        Maximum number of tokens in the context window (sequence length).
    dropout : float
        Dropout probability applied to the attention weights (helps regularization).
    qkv_bias : bool, optional (default=False)
        If True, adds a learnable bias term to the query, key, and value projections.

    Attributes
    ----------
    W_query : nn.Linear
        Linear projection layer mapping input embeddings to query vectors.
    W_key : nn.Linear
        Linear projection layer mapping input embeddings to key vectors.
    W_value : nn.Linear
        Linear projection layer mapping input embeddings to value vectors.
    dropout : nn.Dropout
        Dropout layer applied to attention weights.
    mask : torch.Tensor
        Upper-triangular binary mask (with ones above the diagonal).
        Used to enforce the causal constraint in attention.

    Notes
    -----
    - Attention scores are computed as:
        `scores = Q @ K^T / sqrt(d_k)`
    - The causal mask sets all future positions to `-inf` before softmax,
      preventing tokens from attending to future tokens.
    - This is a building block for Transformers used in GPT-like models.

    Examples
    --------
    >>> attn = CausalSelfAttention(16, 16, context_length=10, dropout=0.1)
    >>> x = torch.randn(2, 10, 16)  # batch of 2 sequences, length 10, embedding dim 16
    >>> out = attn(x)
    >>> out.shape
    torch.Size([2, 10, 16])
    """

    def __init__(self, input_dimension, output_dimension, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.output_dimension = output_dimension
        self.W_query = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.W_key   = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.W_value = nn.Linear(input_dimension, output_dimension, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # Register mask as a persistent buffer (not a parameter, not trainable).
        # "1"s above the diagonal → positions to block.
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Forward pass of causal self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_tokens, input_dimension).

        Returns
        -------
        context_vec : torch.Tensor
            Contextualized embeddings of shape (batch_size, n_tokens, output_dimension).
        """
        b, n_tokens, input_dimension = x.shape

        # Project inputs into queries, keys, and values
        keys = self.W_key(x)       # (b, n_tokens, d_out)
        queries = self.W_query(x)  # (b, n_tokens, d_out)
        values = self.W_value(x)   # (b, n_tokens, d_out)

        # Scaled dot-product attention
        attn_scores = queries @ keys.transpose(1, 2)  # (b, n_tokens, n_tokens)

        # Apply causal mask (future tokens → -inf)
        attn_scores.masked_fill_(
            self.mask.bool()[:n_tokens, :n_tokens], -torch.inf
        )

        # Normalize scores into probabilities
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context_vec = attn_weights @ values  # (b, n_tokens, d_out)

        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    """
    A wrapper around multiple `CausalSelfAttention` heads, implementing 
    **multi-head attention**.

    Each head independently attends to different parts of the input sequence,
    and their outputs are concatenated and projected back into the model dimension.

    Parameters
    ----------
    input_dimension : int
        Dimensionality of the input embeddings.
    output_dimension : int
        Dimensionality of each head's output.
    context_length : int
        Maximum number of tokens in the context window.
    dropout : float
        Dropout probability applied inside each attention head.
    num_heads : int
        Number of parallel attention heads.
    qkv_bias : bool, optional (default=False)
        Whether to include biases in query, key, and value projections.

    Attributes
    ----------
    heads : nn.ModuleList
        List of `CausalSelfAttention` modules.
    out_proj : nn.Linear
        Final linear projection mapping concatenated heads back to output space.

    Examples
    --------
    >>> mha = MultiHeadAttentionWrapper(16, 8, context_length=10, dropout=0.1, num_heads=4)
    >>> x = torch.randn(2, 10, 16)
    >>> out = mha(x)
    >>> out.shape
    torch.Size([2, 10, 32])
    """

    def __init__(self, input_dimension, output_dimension, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalSelfAttention(input_dimension, output_dimension, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )
        # self.out_proj = nn.Linear(output_dimension*num_heads, output_dimension*num_heads)

    def forward(self, x):
        """
        Forward pass of multi-head attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_tokens, input_dimension).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_tokens, output_dimension * num_heads).
        """
        # Run each head independently and concatenate their outputs
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]
dropout=0.0
num_heads=2
d_in, d_out = 3, 2

mha = MultiHeadAttentionWrapper(
    d_in,
    d_out,
    context_length,
    0.0,
    num_heads
)

out = mha(batch)
print("Multi head attention", out)
