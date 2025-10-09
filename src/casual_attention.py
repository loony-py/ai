# Documentation: Causal Attention Implementation in PyTorch
# Overview

# This module provides an implementation of a simplified Causal Attention Mechanism using PyTorch.
# The code demonstrates the key steps involved in computing attention, applying masks for causality, and normalizing attention weights with optional dropout.

# Causal attention is widely used in transformer architectures, particularly in autoregressive models (like GPT),
#  where each token can only attend to its own past and not to the future.

import torch
import torch.nn as nn
from data_inputs import inputs

# Input dimension is derived from the shape of the input tensor (embedding size per token)
input_embedding_dimension = inputs.shape[1]  # Example: 2
# Output dimension is the dimensionality of the transformed embedding space
output_embedding_dimension = 2  # Example: 3


class CasualAttention(nn.Module):
    """
    Implements a simple causal (masked) self-attention mechanism.

    This module projects the input sequence into queries, keys, and values,
    computes scaled dot-product attention scores, applies softmax normalization,
    and produces context vectors. A causal mask is applied externally to ensure
    that tokens only attend to themselves and previous tokens.
    
    Attributes:
        output_embedding_dimension (int): Dimensionality of query/key/value embeddings.
        W_query (nn.Linear): Linear transformation for generating queries.
        W_key (nn.Linear): Linear transformation for generating keys.
        W_value (nn.Linear): Linear transformation for generating values.
        dropout (nn.Dropout): Dropout applied to attention weights.
        mask (torch.Tensor): Upper-triangular mask to enforce causal attention.
    """

    def __init__(
        self,
        input_embedding_dimension: int,
        output_embedding_dimension: int,
        context_length: int,
        dropout_probability: float,
        qkv_bias: bool = False
    ):
        """
        Initializes the Causal Attention layer.
        
        Args:
            input_embedding_dimension (int): Dimensionality of the input token embeddings.
            output_embedding_dimension (int): Dimensionality of the output embeddings 
                                              for query, key, and value vectors.
            context_length (int): Maximum sequence length (used to create the causal mask).
            dropout_probability (float): Dropout probability applied to attention weights.
            qkv_bias (bool, optional): Whether to include bias in the linear projections.
                                       Defaults to False.
        """
        super().__init__()
        self.output_embedding_dimension = output_embedding_dimension

        # Linear projections for queries, keys, and values
        self.W_query = nn.Linear(input_embedding_dimension, output_embedding_dimension, bias=qkv_bias)
        self.W_key = nn.Linear(input_embedding_dimension, output_embedding_dimension, bias=qkv_bias)
        self.W_value = nn.Linear(input_embedding_dimension, output_embedding_dimension, bias=qkv_bias)

        # Dropout applied after softmax
        self.dropout = nn.Dropout(dropout_probability)

        # The use of register_buffer in PyTorch is not strictly necessary for all use cases but offers several advantages here. 
        # For instance, when we use the CausalAttention class in our LLM, buffers are automatically moved to the appropriate 
        # device (CPU or GPU) along with our model, which will be relevant when training our LLM. 
        # This means we don’t need to manually ensure these tensors are on the same device as your model parameters, avoiding device mismatch errors.
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Causal Attention mechanism.
        
        Args:
            input_tensor (torch.Tensor): Shape (batch_size, sequence_length, input_embedding_dimension).
        
        Returns:
            torch.Tensor: Contextualized token representations 
                          (batch_size, sequence_length, output_embedding_dimension).
        """
        batch_size, sequence_length, input_embedding_dimension = input_tensor.shape

        # Compute keys, queries, and values
        keys = self.W_key(input_tensor)       # (batch_size, sequence_length, output_dim)
        queries = self.W_query(input_tensor)  # (batch_size, sequence_length, output_dim)
        values = self.W_value(input_tensor)   # (batch_size, sequence_length, output_dim)

        # Compute raw attention scores (dot product of queries and keys)
        attention_scores = queries @ keys.transpose(1, 2)  # (batch_size, sequence_length, sequence_length)
        print(f"Attention scores: {attention_scores}")
        # Apply causal mask (prevent looking ahead)
        attention_scores.masked_fill_(
            self.mask.bool()[:sequence_length, :sequence_length],
            float("-inf")
        )
        print(f"Attention scores masked fill: {attention_scores}")

        # Normalize attention scores with softmax and scale by sqrt(d_k)
        attention_weights = torch.softmax(
            attention_scores / (keys.shape[-1] ** 0.5),
            dim=-1
        )
        print(f"Attention weights after applying softmax: {attention_weights}")

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        print(f"Dropped out Attention weights: {attention_weights}")

        # Compute weighted sum of values (context vectors)
        context_vectors = attention_weights @ values  # (batch_size, sequence_length, output_dim)

        return context_vectors


# Example usage
if __name__ == "__main__":
    # Create a batch of size 2 by stacking the same input twice
    batch_input = torch.stack((inputs, inputs), dim=0)
    print(f"Batch input: {batch_input}")
    # Fix random seed for reproducibility
    torch.manual_seed(123)

    # Define maximum sequence length (context length)
    context_length = batch_input.shape[1]
    print(f"Context length: {context_length}")

    # Initialize Causal Attention module
    causal_attention = CasualAttention(
        input_embedding_dimension,
        output_embedding_dimension,
        context_length,
        dropout_probability=0.0
    )

    # Compute context vectors
    context_vectors = causal_attention.forward(batch_input)

    print("Context vectors:", context_vectors)
    print("Context vectors Shape:", context_vectors.shape)
