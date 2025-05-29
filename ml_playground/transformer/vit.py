import torch
from torch import nn
from transformer import EncoderLayer, PositionalEncoding


class MNISTVisionTransformer(nn.Module):
    def __init__(
        self,
        num_patches: int = 7,
        d_model: int = 8,
        num_heads: int = 2,
        num_layers: int = 2,
        d_ff: int = 1024,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.c, self.h, self.w = 1, 28, 28
        self.num_patches = num_patches

        if self.h % num_patches != 0 or self.w % num_patches != 0:
            raise ValueError(
                f"Image size {self.h}x{self.w} must be divisible by num_patches {self.num_patches}."
            )
        self.patch_size = (self.h // self.num_patches, self.w // self.num_patches)

        # Linear mapping
        self.input_d = self.c * self.patch_size[0] * self.patch_size[1]
        self.linear_mapper = nn.Linear(self.input_d, d_model)

        # Learnable classifiation token
        self.class_token = nn.Parameter(torch.rand(1, d_model))

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, self.h * self.w)

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(d_model, num_classes), nn.Softmax(dim=-1)
        )

    def patchify(self, x: torch.Tensor, num_patches: int = 7) -> torch.Tensor:
        """Convert images into patches.

        Break each (1, 28, 28) image into 7x7 patches (hence, each of size 4x4)
        and obtain 7x7=49 sub-images out of a single image.
        Each patch is a picture of size 1x4x4 which is flattened to a 16-dimensional vector.

        Args:
            x (torch.Tensor): MNIST images of [N, C=1, H=28, W=28]
            num_patches (int): Number of patches (default is 7)

        Returns:
            torch.Tensor: Output tensor of shape [N, 7*7=49, 4*4*1=16]
        """
        n, c, h, w = x.shape

        if h != w:
            raise ValueError(f"Expected square images, got {h}x{w}.")

        # Reshape the input tensor to extract patches
        # [N, C, H, W] -> [N, C, num_patches, num_patches, patch_size, patch_size]
        x = x.unfold(2, self.patch_size[0], self.patch_size[1])
        x = x.unfold(3, self.patch_size[0], self.patch_size[1])
        # Reorder dimensions [N, num_patches, num_patches, C, patch_size, patch_size]
        patches = x.permute(0, 2, 3, 1, 4, 5)
        # Reshape patches to [N, 7*7=49, C*4*4=16]
        patches = patches.reshape(
            n, num_patches * num_patches, c * self.patch_size[0] * self.patch_size[1]
        )
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(x, self.num_patches)  # [N, 1, 28, 28] -> [N, 49, 16]
        tokens = self.linear_mapper(patches)  # [N, 49, 16] -> [N, 49, d_model]

        # Create N classification tokens [1, d_model] -> [N, 1, d_model]
        class_token = self.class_token.expand(tokens.shape[0], -1).unsqueeze(1)
        # Add classification token to the image tokens [N, 49+1, d_model]
        tokens = torch.cat((class_token, tokens), dim=1)

        # Add positional encoding to the tokens
        tokens = self.positional_encoding(tokens)

        output = tokens
        for layer in self.layers:
            output = layer(output)

        # Pass only the classification token to the classifier
        return self.classifier(output[:, 0])


class MNISTBaseTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()

        self.embedding = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, 28 * 28)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, num_classes), nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process MNIST images through the transformer model."""
        # Flatten the image into a sequence [N, 1, 28, 28] -> [N, 28*28, 1]
        x = x.view(x.size(0), -1, 1)
        img_embedding = self.dropout(self.positional_encoding(self.embedding(x)))

        output = img_embedding
        for layer in self.layers:
            output = layer(output)

        # Compute mean over pixel features [N, 28*28, d_model] -> [N, d_model]
        # Apply linear layer to get final output [N, d_model] -> [N, num_classes]
        return self.classifier(output.mean(dim=1))
