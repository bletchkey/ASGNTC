import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from .utils import constants as constants

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, E, H/P, W/P]
        x = x.flatten(2)  # [B, E, N]
        x = x.transpose(1, 2)  # [B, N, E]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)

        # Linear projections
        query = self.query(x).view(batch_size, -1, self.num_heads, self.attention_head_size).transpose(1, 2)
        key = self.key(x).view(batch_size, -1, self.num_heads, self.attention_head_size).transpose(1, 2)
        value = self.value(x).view(batch_size, -1, self.num_heads, self.attention_head_size).transpose(1, 2)

        # Attention
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.all_head_size)

        # Output
        output = self.out(context)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=16, in_channels=1, embed_dim=768, num_heads=12, num_layers=12, ff_dim=3072):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(1, 1 + self.patch_embedding.n_patches, embed_dim))
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, ff_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, embed_dim)  # Example output layer

    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Duplicate cls token for batch
        x = self.patch_embedding(x)
        x = torch.cat((cls_tokens, x), dim=1)  # Concatenate cls token with patch embeddings
        x += self.positional_embedding  # Add positional embeddings
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]  # We only use the cls token for the final classification

        # Example: output layer for further processing, e.g., linear layer for classification
        # In a real-world scenario, you might adjust this part depending on your specific task.
        x = self.head(cls_token_final)

        return x


def Predictor_ViT():
    img_size = constants.grid_size
    patch_size = 8  # Example: using 8x8 patches
    in_channels = constants.nc
    embed_dim = 768  # Size of each embedding
    num_heads = 12  # Number of attention heads
    num_layers = 12  # Number of transformer blocks
    ff_dim = 2048  # Dimension of the feedforward network

    model = VisionTransformer(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                          embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim)


    return model