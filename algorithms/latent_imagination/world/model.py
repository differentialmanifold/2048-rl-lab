"""Continuous board latents with a learned discrete chance distribution.

No board rules, future observations, or decoding are used by latent_step.
The parameter layout and serialized configuration preserve existing checkpoints.
"""
from copy import deepcopy

from torch import nn

from ..transformer import blocks, WORLD_TRANSFORMER_VERSION


def mlp(inputs, width, outputs):
    return nn.Sequential(nn.Linear(inputs, width), nn.LayerNorm(width), nn.SiLU(),
                         nn.Linear(width, outputs))


class Tokenizer(nn.Module):
    """Current neural world: fixed 2D RoPE and SwiGLU, no architecture switch."""
    def __init__(self, width, latent_dim, layers, heads, tile_classes):
        super().__init__()
        self.embedding = nn.Embedding(tile_classes, width)
        self.encoder = blocks(width, heads, layers)
        self.compress = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, latent_dim), nn.Tanh())
        self.expand = nn.Linear(latent_dim, width)
        self.decoder = blocks(width, heads, layers)
        self.readout = nn.Linear(width, tile_classes)

    def encode(self, ranks):
        return self.compress(self.encoder(self.embedding(ranks.long())))

    def decode(self, latent):
        return self.readout(self.decoder(self.expand(latent)))


class WorldModel(nn.Module):
    def __init__(self, width=128, latent_dim=16, tokenizer_layers=2, dynamics_layers=2,
                 heads=4, events=33, tile_classes=32, model_version=2, architecture='latent_transformer',
                 transition_kind=None, transformer_version=WORLD_TRANSFORMER_VERSION,state_head_version=1):
        super().__init__()
        if model_version != 2 or architecture != 'latent_transformer':
            raise ValueError('Unsupported latent world model version/architecture')
        if (min(width, latent_dim, tokenizer_layers, dynamics_layers, heads) < 1
                or width % heads or events < 2 or tile_classes < 2):
            raise ValueError('Invalid latent model dimensions')
        self.architecture = architecture
        self.model_config = dict(model_type='latent_dreamer', model_version=model_version,
            architecture=architecture, width=width, latent_dim=latent_dim,
            tokenizer_layers=tokenizer_layers, dynamics_layers=dynamics_layers,
            heads=heads, events=events, tile_classes=tile_classes)
        if model_version == 2:
            if transformer_version != WORLD_TRANSFORMER_VERSION:
                raise ValueError('Neural world requires the current 2D RoPE/SwiGLU architecture')
            self.model_config['transformer_version'] = WORLD_TRANSFORMER_VERSION
            if transition_kind not in (None, 'linear_residual'):
                raise ValueError('Unsupported neural transition parameterization')
            self.model_config['transition_kind'] = 'linear_residual'
            if state_head_version not in (1,2,3): raise ValueError('Unsupported state head version')
            if state_head_version>=2: self.model_config['state_head_version']=state_head_version
        from .dynamics import NeuralWorld
        self.world = NeuralWorld(width, latent_dim, tokenizer_layers, dynamics_layers, heads, events,
                                 tile_classes, state_head_version=state_head_version)
        # Inactive policy tensors keep the state_dict of existing v2 worlds
        # loadable. Latent PPO has its own separate policy module.
        self.actor = mlp(16 * latent_dim, width, 4)
        self.critic = mlp(16 * latent_dim, width, 1)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)
        self.policy_prior = deepcopy(self.actor)
        self.slow_critic = deepcopy(self.critic)
        self.phase = None
        self.policy_prior.requires_grad_(False)
        self.slow_critic.requires_grad_(False)

    def set_phase(self, phase):
        self.requires_grad_(False)
        if phase == 'tokenizer':
            self.world.tokenizer.requires_grad_(True)
        elif phase == 'world':
            self.world.requires_grad_(True)
            self.world.tokenizer.requires_grad_(False)
            if getattr(self.world,'state_head_version',1)>=3:
                self.world.legality.requires_grad_(False)  # Legacy weights retained for checkpoint migration.
        else:
            raise ValueError(f'Unknown phase: {phase}')
        self.phase = phase
