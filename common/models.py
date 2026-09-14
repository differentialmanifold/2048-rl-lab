"""Shared MLP and ResCNN; parameter names preserve existing v3 checkpoints."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REWARD_OBJECTIVE = "spawn_mass"
REWARD_SCALE = 128.0


def preprocess_observation(obs):
    flat = np.asarray(obs, dtype=np.float32).reshape(-1)
    exponents = np.zeros_like(flat)
    np.log2(flat, out=exponents, where=flat > 0)
    return torch.from_numpy(exponents)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels))

    def forward(self, x):
        return F.silu(x + self.layers(x))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=16, num_actions=4, architecture='mlp'):
        super().__init__()
        self.obs_dim = obs_dim
        self.model_config = dict(obs_dim=obs_dim, num_actions=num_actions, architecture=architecture)
        self.architecture = architecture
        self.embedding = nn.Embedding(32, 16)
        if architecture == 'mlp':
            # Preserve parameter names/shapes to resume existing v3 checkpoints exactly.
            self.trunk = nn.Sequential(nn.Linear(obs_dim * 16, 256), nn.ReLU(),
                                       nn.Linear(256, 256), nn.ReLU())
        elif architecture == 'rescnn':
            if obs_dim != 16:
                raise ValueError('rescnn currently supports the 4x4 board only')
            self.trunk = nn.Sequential(
                nn.Conv2d(17, 32, 3, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
                ResidualConvBlock(), ResidualConvBlock(), nn.Flatten(),
                nn.Linear(32 * 16, 256), nn.LayerNorm(256), nn.SiLU())
        else:
            raise ValueError(f'Unknown model architecture: {architecture}')
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def features(self, x):
        embeddings = self.embedding(x.long().clamp(0, 31))
        if self.architecture == 'mlp':
            return self.trunk(embeddings.flatten(-2))
        # Categorical identity plus numeric rank; retain spatial positions on the board.
        single = x.ndim == 1
        image = embeddings.reshape(-1, 4, 4, 16).permute(0, 3, 1, 2)
        ranks = x.reshape(-1, 1, 4, 4).float() / 16.0
        features = self.trunk(torch.cat((image, ranks), dim=1))
        return features[0] if single else features

    def forward(self, x):
        features = self.features(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


def masked_categorical(logits, legal_mask):
    """One differentiable distribution for sampling, old/new log-p and entropy."""
    mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=logits.device)
    mask = torch.broadcast_to(mask, logits.shape)
    if not mask.any(dim=-1).all():
        raise ValueError('Cannot construct a policy for a terminal/all-invalid state')
    return torch.distributions.Categorical(
        logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min))
