"""Spatial attention for the neural world and its separate latent policy."""
from torch import nn
from torch.nn import functional as F

from common.models import RotaryPosition2D

WORLD_TRANSFORMER_VERSION = 2


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        # Three matrices instead of two: approximate the legacy 2*width FFN
        # parameter budget, rounding the hidden width up to a multiple of eight.
        hidden = ((4 * width + 23) // 24) * 8
        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, width)

    def forward(self, x):
        return self.output(F.silu(self.gate(x)) * self.value(x))


class SpatialTransformerBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads or (width // heads) % 4:
            raise ValueError('2D RoPE requires width/heads to be divisible by four')
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.rotary = RotaryPosition2D(head_dim=width // heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = SwiGLU(width)

    def forward(self, x):
        batch, tokens, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, tokens, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            self.rotary(query), self.rotary(key), value, dropout_p=0., is_causal=False)
        x = x + self.attention_output(attended.transpose(1, 2).reshape(batch, tokens, width))
        return x + self.mlp(self.norm2(x))


def blocks(width, heads, layers):
    return nn.Sequential(*[SpatialTransformerBlock(width, heads) for _ in range(layers)])
