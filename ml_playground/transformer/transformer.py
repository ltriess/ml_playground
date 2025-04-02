import math
from typing import Tuple

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class Transformer(nn.Module):
    """The transformer architecture.

    1. Generate source and target masks using the generate_mask method.
    2. Compute source and target embeddings, and apply positional encoding and dropout.
    3. Process the source sequence through encoder layers, updating the enc_output tensor.
    4. Process the target sequence through decoder layers, using enc_output and masks,
       and updating the dec_output tensor.
    5. Apply the linear projection layer to the decoder output, obtaining output logits.
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        super().__init__()

        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)
        ).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedding = self.dropout(
            self.positional_encoding(self.encoder_embedding(src))
        )
        tgt_embedding = self.dropout(
            self.positional_encoding(self.decoder_embedding(tgt))
        )

        enc_output = src_embedding
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedding
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        return self.linear(dec_output)


if __name__ == "__main__":
    SRC_VOCAB_SIZE = 5000
    TGT_VOCAB_SIZE = 5000

    transformer = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_length=100,
        dropout=0.1,
    )

    # Generate random sample data of shape (batch_size, seq_length)
    src_data = torch.randint(1, SRC_VOCAB_SIZE, (64, 100))
    tgt_data = torch.randint(1, TGT_VOCAB_SIZE, (64, 100))

    # "Train" the model
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9
    )

    transformer.train()

    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        loss = criterion(
            output.contiguous().view(-1, TGT_VOCAB_SIZE),
            tgt_data[:, 1:].contiguous().view(-1),
        )
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1:4d}    Loss: {loss.item():>6f}")
