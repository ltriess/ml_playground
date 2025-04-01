import math

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention."""

    def __init__(self, d_k):
        super().__init__()

        self.d_k = d_k

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """Computes the scaled dot-product attention.

        Args:
            q: torch.Tensor         The query tensor of shape (batch_size, seq_length, d_k).
            k: torch.Tensor         The key tensor of shape (batch_size, seq_length, d_k).
            v: torch.Tensor         The value tensor of shape (batch_size, seq_length, d_k).
            mask: torch.Tensor      Optional mask tensor of shape (batch_size, 1, seq_length).

        Returns:
            torch.Tensor            The output of the scaled dot-product attention.
        """
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_probabilities = torch.softmax(attention_scores, dim=-1)

        return torch.matmul(attention_probabilities, v)


class MultiHeadAttention(nn.Module):
    """This module implements the multi-head attention mechanism used in transformers."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.scaled_dot_product_attehtion = ScaledDotProductAttention(d_k=self.d_k)

    def split_heads(self, x):
        """Splits the input tensor into multiple heads.

        Example:
            input shape: (2, 10, 512) => output shape (2, 8, 10, 64)

        Args:
            x: torch.Tensor     The input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor        The output tensor of shape (batch_size, num_heads, seq_length, d_k).
        """
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """Combines the heads of the multi-head attention output.

        Example:
            input shape: (2, 8, 10, 64) => output shape (2, 10, 512)

        Args:
            x: torch.Tensor     The input tensor of shape (batch_size, num_heads, seq_length, d_k).

        Returns:
            torch.Tensor        The output tensor of shape (batch_size, seq_length, d_model).
        """
        batch_size, _, seq_length, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """Computes the multi-head attention.

        Args:
            q: torch.Tensor         The query tensor of shape (batch_size, seq_length, d_model).
            k: torch.Tensor         The key tensor of shape (batch_size, seq_length, d_model).
            v: torch.Tensor         The value tensor of shape (batch_size, seq_length, d_model).
            mask: torch.Tensor      Optional mask tensor of shape (batch_size, 1, seq_length).

        Returns:
            torch.Tensor            The output of the multi-head attention mechanism.
        """
        q = self.split_heads(self.w_q(q))
        k = self.split_heads(self.w_k(k))
        v = self.split_heads(self.w_v(v))

        attention_output = self.scaled_dot_product_attehtion(q, k, v, mask)
        return self.w_o(self.combine_heads(attention_output))


if __name__ == "__main__":
    model = MultiHeadAttention(d_model=512, num_heads=8)
    print(model)

    q = torch.rand(2, 10, 512)  # (batch_size, seq_length, d_model)
    k = torch.rand(2, 10, 512)  # (batch_size, seq_length, d_model)
    v = torch.rand(2, 10, 512)  # (batch_size, seq_length, d_model)

    output = model(q, k, v)
    print(output.size())  # (batch_size, seq_length, d_model)
