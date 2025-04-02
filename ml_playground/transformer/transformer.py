import math

import torch
from attention import MultiHeadAttention
from torch import nn


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward network.

    Enables the model to consider the position of input elements while making predictions.
    """

    def __init__(self, d_model, d_ff):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    """Positional Encoding.

    Inject position information of each token into the input sequence.
    It uses sine and cosine functions of different frequencies to encode the position.
    """

    def __init__(self, d_model, max_seq_length):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    """Transofmer encoder layer."""

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attention_output = self.self_attention(x, x, x, mask=mask)
        x = self.norm_1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm_2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    """Transformer decoder layer.

    1. Calculate the masked self-attention output and add it to the input tensor,
       followed by dropout and layer normalization.
    2. Compute the cross-attention output between the decoder and encoder outputs, and add it
       to the normalized masked self-attention output, followed by dropout and layer normalization.
    3. Calculate the position-wise feed-forward output and combine it with the normalized
       cross-attention output, followed by dropout and layer normalization.
    4. Return the processed tensor.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        attention_output = self.self_attention(x, x, x, mask=tgt_mask)
        x = self.norm_1(x + self.dropout(attention_output))
        attention_output = self.cross_attention(
            x, enc_output, enc_output, mask=src_mask
        )
        x = self.norm_2(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm_3(x + self.dropout(ff_output))
        return x


if __name__ == "__main__":
    positional_encoding = PositionalEncoding(d_model=512, max_seq_length=1000)
    encoder_layer = EncoderLayer(d_model=512, num_heads=8, d_ff=128, dropout=0.5)
    decoder_layer = DecoderLayer(d_model=512, num_heads=8, d_ff=128, dropout=0.5)

    print(positional_encoding)
    print(encoder_layer)
    print(decoder_layer)
