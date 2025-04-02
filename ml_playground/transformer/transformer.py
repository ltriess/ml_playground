import torch
from attention import MultiHeadAttention

if __name__ == "__main__":
    model = MultiHeadAttention(d_model=512, num_heads=8)
    print(model)

    q = torch.rand(2, 10, 512)  # (batch_size, seq_length, d_model)
    k = torch.rand(2, 10, 512)  # (batch_size, seq_length, d_model)
    v = torch.rand(2, 10, 512)  # (batch_size, seq_length, d_model)

    output = model(q, k, v)
    print(output.size())  # (batch_size, seq_length, d_model)
