"""Shared board encoders and actor/critic heads; existing checkpoint keys are preserved."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REWARD_OBJECTIVE = "spawn_mass"
REWARD_SCALE = 128.0
ARCHITECTURES = ('cnn2x2', 'vit')
CNN2X2_ENCODER_VERSION = 3  # Two valid convolutions, 64 -> 128, with a projected residual.
VIT_ENCODER_VERSION = 2  # Ordered cell readout and 2D rotary position encoding.


def preprocess_observation(obs):
    flat = np.asarray(obs, dtype=np.float32).reshape(-1)
    exponents = np.zeros_like(flat)
    np.log2(flat, out=exponents, where=flat > 0)
    return torch.from_numpy(exponents)


class ValidResidualConvBlock(nn.Module):
    """Two unpadded 2x2 convolutions: 17x4x4 -> 64x3x3 -> 128x2x2.

    Main path: conv/norm/SiLU, conv/norm, add shortcut, SiLU.
    The shortcut pools over the same 3x3 receptive field and projects channels
    so the shrinking spatial dimensions can be added without padding/cropping.
    """
    def __init__(self, in_channels=17, hidden_channels=64, out_channels=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 2), nn.GroupNorm(4, hidden_channels), nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, 2), nn.GroupNorm(4, out_channels))
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(3, stride=1),
            nn.Conv2d(in_channels, out_channels, 1), nn.GroupNorm(4, out_channels))

    def forward(self, x):
        return F.silu(self.shortcut(x) + self.layers(x))


class TransformerBlock(nn.Module):
    """Pre-norm, bidirectional attention over board tokens; no dropout."""
    def __init__(self, width=96, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_output = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                 nn.Linear(2 * width, width))

    def forward(self, x, rotary=None):
        batch, tokens, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, tokens, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if rotary is not None:
            query, key = rotary(query), rotary(key)
        attended = F.scaled_dot_product_attention(query, key, value,
                                                  dropout_p=0., is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, tokens, width)
        x = x + self.attention_output(attended)
        return x + self.mlp(self.norm2(x))


class RotaryPosition2D(nn.Module):
    """Axial RoPE: rotate the first half of each head by row, the rest by col.

    Each axis rotates adjacent feature pairs. Q and K use the same rotation,
    making their inner product depend on relative row/column displacements.
    Real-valued operations keep this compatible with CPU, CUDA and MPS.
    """
    def __init__(self, head_dim=24, size=4, base=10000.):
        super().__init__()
        if head_dim <= 0 or head_dim % 4:
            raise ValueError('2D RoPE head dimension must be positive and divisible by four')
        axis_dim = head_dim // 2
        frequencies = base ** (-torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim)
        coordinates = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size),
                                                 indexing='ij'), dim=-1).reshape(-1, 2)
        angles = torch.cat((coordinates[:, :1] * frequencies,
                            coordinates[:, 1:] * frequencies), dim=-1)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def forward(self, x):
        even, odd = x[..., 0::2], x[..., 1::2]
        cos, sin = self.cos.to(dtype=x.dtype), self.sin.to(dtype=x.dtype)
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class BoardViT(nn.Module):
    """Keep every cell at readout and apply row/column RoPE to attention Q/K.

    A 96-wide, two-block, four-head transformer. No CLS or
    absolute position embedding: ordered readout gives absolute cell identity,
    while RoPE supplies relative geometry in every block. Compress each cell to
    eight features before flattening to keep the parameter budget compact.
    """
    def __init__(self):
        super().__init__()
        self.token_projection = nn.Linear(17, 96)
        self.blocks = nn.ModuleList([TransformerBlock(), TransformerBlock()])
        self.rotary = RotaryPosition2D()
        self.norm = nn.LayerNorm(96)
        self.cell_projection = nn.Linear(96, 8)
        self.projection = nn.Sequential(nn.Linear(16 * 8, 256), nn.GELU())

    def forward(self, tiles):
        tokens = self.token_projection(tiles)
        for block in self.blocks:
            tokens = block(tokens, self.rotary)
        cells = self.cell_projection(self.norm(tokens))
        return self.projection(cells.flatten(1))


class BoardEncoder(nn.Module):
    """Shared board -> 256 features, without algorithm-specific output heads."""
    def __init__(self, obs_dim=16, num_actions=4, architecture='cnn2x2'):
        super().__init__()
        self.obs_dim = obs_dim
        self.model_config = dict(obs_dim=obs_dim, num_actions=num_actions, architecture=architecture)
        if architecture == 'cnn2x2':
            self.model_config['encoder_version'] = CNN2X2_ENCODER_VERSION
        elif architecture == 'vit':
            self.model_config['encoder_version'] = VIT_ENCODER_VERSION
        self.architecture = architecture
        if obs_dim != 16:
            raise ValueError(f'{architecture} currently supports the 4x4 board only')
        self.embedding = nn.Embedding(32, 16)
        if architecture == 'cnn2x2':
            self.trunk = nn.Sequential(
                ValidResidualConvBlock(), nn.Flatten(),
                nn.Linear(128 * 2 * 2, 256), nn.LayerNorm(256), nn.SiLU())
        elif architecture == 'vit':
            self.trunk = BoardViT()
        else:
            raise ValueError(f'Unknown model architecture: {architecture}')

    def features(self, x):
        embeddings = self.embedding(x.long().clamp(0, 31))
        # Categorical identity plus numeric rank; retain spatial positions on the board.
        single = x.ndim == 1
        if self.architecture == 'vit':
            tiles = torch.cat((embeddings.reshape(-1, 16, 16),
                               x.reshape(-1, 16, 1).float() / 16.0), dim=-1)
            features = self.trunk(tiles)
            return features[0] if single else features
        image = embeddings.reshape(-1, 4, 4, 16).permute(0, 3, 1, 2)
        ranks = x.reshape(-1, 1, 4, 4).float() / 16.0
        features = self.trunk(torch.cat((image, ranks), dim=1))
        return features[0] if single else features


class ActorCritic(BoardEncoder):
    def __init__(self, obs_dim=16, num_actions=4, architecture='cnn2x2'):
        super().__init__(obs_dim, num_actions, architecture)
        # Keep names, shapes and initialization order of existing checkpoints.
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x):
        features = self.features(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


def validate_encoder_version(config):
    if config.get('architecture') not in ARCHITECTURES:
        raise ValueError('Checkpoint requires a supported architecture: cnn2x2 or vit')
    if (config.get('architecture') == 'cnn2x2'
            and config.get('encoder_version') != CNN2X2_ENCODER_VERSION):
        raise ValueError('Incompatible cnn2x2 encoder version: the current unpadded residual CNN2x2 requires '
                         'a new training run; previous CNN2x2 versions cannot be loaded or resumed')
    if (config.get('architecture') == 'vit'
            and config.get('encoder_version') != VIT_ENCODER_VERSION):
        raise ValueError('Incompatible vit encoder version: ordered cell readout and 2D RoPE '
                         'require a new training run; previous CLS ViT checkpoints cannot be loaded or resumed')


def model_from_config(config):
    """Serialization dispatch only; each algorithm owns its training logic."""
    config = dict(config)
    validate_encoder_version(config)
    config.pop('encoder_version', None)
    model_type = config.pop('model_type', 'actor_critic')
    if model_type == 'actor_critic':
        return ActorCritic(**config)
    if model_type == 'muzero':
        from algorithms.muzero import MuZeroNetwork
        return MuZeroNetwork(**config)
    raise ValueError(f'Unknown checkpoint model type: {model_type}')


def masked_categorical(logits, legal_mask):
    """One differentiable distribution for sampling, old/new log-p and entropy."""
    mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=logits.device)
    mask = torch.broadcast_to(mask, logits.shape)
    if not mask.any(dim=-1).all():
        raise ValueError('Cannot construct a policy for a terminal/all-invalid state')
    return torch.distributions.Categorical(
        logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min))
