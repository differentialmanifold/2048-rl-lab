"""Shared spatial ViT or CNN2x2 actor/critic on continuous afterstate latents."""
from copy import deepcopy

import torch
from torch import nn

from .world.model import WorldModel
from .transformer import blocks
from common.models import ValidResidualConvBlock


class ImaginationPolicy(nn.Module):
    """Spatial ViT or valid 2x2 residual CNN on latents; no board decoder.

    Ordered per-cell readout retains board orientation. All four candidate
    afterstates share this trainable trunk and the scalar policy/value heads.
    """
    def __init__(self, latent_dim, architecture='vit'):
        super().__init__()
        self.latent_dim = latent_dim
        self.architecture = architecture
        if architecture == 'vit':
            # Keep existing ViT parameter names and initialization order.
            self.input_projection = nn.Linear(latent_dim, 96)
            self.blocks = blocks(96, 4, 2)
            self.norm = nn.LayerNorm(96)
            self.cell_projection = nn.Linear(96, 8)
            self.readout = nn.Sequential(nn.Flatten(), nn.Linear(16 * 8, 256), nn.LayerNorm(256), nn.SiLU())
        elif architecture == 'cnn2x2':
            self.trunk = nn.Sequential(ValidResidualConvBlock(in_channels=latent_dim),
                nn.Flatten(), nn.Linear(128 * 2 * 2, 256), nn.LayerNorm(256), nn.SiLU())
        else:
            raise ValueError('Expected vit or cnn2x2 policy')
        self.policy_head = nn.Linear(256, 1)
        self.value_head = nn.Linear(256, 1)
        nn.init.orthogonal_(self.policy_head.weight, gain=.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, candidates):
        if candidates.ndim not in (3, 4) or candidates.shape[-3:] != (4, 16, self.latent_dim):
            raise ValueError('Expected [B,4,16,L] or [4,16,L] candidate afterstate latents')
        shape = candidates.shape[:-2]
        if self.architecture == 'vit':
            cells = self.input_projection(candidates.reshape(-1, 16, self.latent_dim))
            cells = self.cell_projection(self.norm(self.blocks(cells)))
            features = self.readout(cells)
        else:
            cells = candidates.reshape(-1, 4, 4, self.latent_dim).permute(0, 3, 1, 2)
            features = self.trunk(cells)
        return self.policy_head(features).reshape(shape), self.value_head(features).reshape(shape)


class ImaginationAgent(nn.Module):
    display_name = "Latent Imagination RL"

    def __init__(self, world_config, architecture='vit', policy_version=1):
        super().__init__()
        if architecture not in ('vit', 'cnn2x2') or policy_version != 1:
            raise ValueError('Imagination policy requires ViT/CNN2x2 policy version 1')
        config = deepcopy(world_config)
        if config.pop('model_type', None) != 'latent_dreamer' or config.get('model_version') != 2:
            raise ValueError('Imagination policy needs a neural v2 world')
        self.world = WorldModel(**config).world
        self.world.requires_grad_(False)
        self.world.eval()
        self.policy = ImaginationPolicy(config['latent_dim'], architecture)
        self.architecture = architecture
        self.model_config = dict(model_type='latent_afterstate_ppo', architecture=architecture,
                                 policy_version=policy_version, world_config=deepcopy(world_config))

    def train(self, mode=True):
        super().train(mode)
        self.world.eval()
        return self

    @torch.no_grad()
    def candidate_latents(self, z):
        actions = torch.arange(4, device=z.device).repeat(len(z))
        states = z[:, None].expand(-1, 4, -1, -1).reshape(-1, *z.shape[1:])
        return self.world.afterstate(states, actions).reshape(len(z), 4, *z.shape[1:])

    def forward(self, ranks):
        """Real-environment inference: encode the observation, then learned F.

        Return four action scores and four afterstate values. Evaluation uses
        the real environment's legal mask, just as the reference PPO does.
        """
        single = ranks.ndim == 1
        with torch.no_grad():
            z = self.world.encode(ranks.reshape(-1, 16))
            candidates = self.candidate_latents(z)
        logits, values = self.policy(candidates)
        return (logits[0], values[0]) if single else (logits, values)
