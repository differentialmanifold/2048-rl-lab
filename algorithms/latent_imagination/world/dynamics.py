"""Learned continuous latent dynamics; no board rules, tables or decoding in steps.

Afterstate/event labels are privileged TRAINING supervision supplied
by a separate data preparation module. Every transition/head at inference is
a neural prediction. All 33 event probabilities are learned without masks.
"""
import torch
import hashlib
from torch import nn

from ..transformer import blocks
from .model import Tokenizer, mlp


def world_fingerprint(world):
    """Stable world-only identity; actor/optimizer changes do not invalidate it."""
    digest = hashlib.sha256()
    for name, value in sorted(world.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}\0'.encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ResidualTransition(nn.Module):
    def __init__(self, width, latent_dim, layers, heads):
        super().__init__()
        self.projection = nn.Linear(latent_dim, width)
        self.blocks = blocks(width, heads, layers)
        self.output = nn.Linear(width, latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent, condition):
        if condition.ndim == 2:
            condition = condition[:, None]
        delta = self.output(self.blocks(self.projection(latent) + condition))
        # Linear residual keeps gradients alive when the pretrained encoder's
        # tanh coordinates saturate. Latent consistency and multi-step fitting
        # learn the state manifold; inference does not clamp/project to rules.
        return latent + delta


class NeuralWorld(nn.Module):
    semantic_events = True

    def __init__(self, width, latent_dim, tokenizer_layers, dynamics_layers, heads, events, tile_classes,state_head_version=1):
        super().__init__()
        if events != 33:
            raise ValueError('Supervised event vocabulary requires 33 labels')
        self.events, self.latent_dim = events, latent_dim
        self.state_head_version = state_head_version
        flat = 16 * latent_dim
        self.tokenizer = Tokenizer(width, latent_dim, tokenizer_layers, heads, tile_classes)
        self.action_embedding = nn.Embedding(4, width)
        # Learned spatial conditioning, with no fixed cell writes or merge rules.
        self.event_embedding = nn.Embedding(events, 16 * width)
        self.action_transition = ResidualTransition(width, latent_dim, dynamics_layers, heads)
        self.event_transition = ResidualTransition(width, latent_dim, dynamics_layers, heads)
        self.prior = mlp(2 * flat + width, width, events)
        self.reward = mlp(2 * flat + width, width, 3)
        self.reward_event_embedding = nn.Embedding(events, width)
        self.terminal = mlp(flat, width, 1)
        self.legality = mlp(flat, width, 4)
        # Versioned for old checkpoint loading. Zero output preserves the old
        # readout exactly at migration, then learns shared spatial features.
        self.state_refiner=ResidualTransition(width,latent_dim,1,heads) if state_head_version>=2 else None
        self.register_buffer('reward_values', torch.tensor([0., 2., 4.]))

    def encode(self, ranks):
        return self.tokenizer.encode(ranks)

    def decode(self, z):
        return self.tokenizer.decode(z)

    def afterstate(self, z, actions):
        return self.action_transition(z, self.action_embedding(actions.long()))

    def chance_prior(self, u, z, actions):
        # Same afterstate can follow a valid or invalid action. The source and
        # action therefore remain available; neither changed/no-op is hard-coded.
        context = torch.cat((z.flatten(1), u.flatten(1), self.action_embedding(actions.long())), -1)
        return self.prior(context)

    def event_probabilities(self, z, u, actions):
        return self.chance_prior(u, z, actions).softmax(-1)

    def event_step(self, u, events):
        condition = self.event_embedding(events.long()).reshape(len(u), 16, -1)
        return self.event_transition(u, condition)

    def all_events(self, u):
        b = len(u)
        expanded = u[:, None].expand(-1, self.events, -1, -1).reshape(-1, 16, self.latent_dim)
        events = torch.arange(self.events, device=u.device).repeat(b)
        return self.event_step(expanded, events).reshape(b, self.events, 16, self.latent_dim)

    def state_heads(self, z):
        original=z
        if self.state_refiner is not None:
            z=self.state_refiner(z,z.new_zeros(len(z),self.action_embedding.embedding_dim))
        terminal=self.terminal(z.flatten(1)).squeeze(-1)
        if self.state_head_version<3:
            return terminal,self.legality(z.flatten(1))
        # A shared learned event marginal replaces two contradictory classifiers.
        # This is not a board-rule mask: F and the 33-way prior are neural; no
        # tile decoding, comparison or probability clipping occurs here.
        b=len(original)
        states=original[:,None].expand(-1,4,-1,-1).reshape(-1,16,self.latent_dim)
        actions=torch.arange(4,device=z.device).repeat(b)
        after=self.afterstate(states,actions)
        logits=self.chance_prior(after,states,actions)
        legal=(torch.logsumexp(logits[:,:32],-1)-logits[:,32]).reshape(b,4)
        return terminal,legal

    def predict_outputs(self, u, events, next_z):
        reward = self.reward(torch.cat((u.flatten(1), self.reward_event_embedding(events.long()), next_z.flatten(1)), -1))
        terminal, legal = self.state_heads(next_z)
        return reward, terminal, legal

    def step_details(self, z, actions):
        u = self.afterstate(z, actions)
        probabilities = self.event_probabilities(z, u, actions)
        events = torch.multinomial(probabilities, 1).squeeze(-1)
        next_z = self.event_step(u, events)
        reward_logits, terminal, legal = self.predict_outputs(u, events, next_z)
        return dict(afterstate=u, probabilities=probabilities, event=events, next_z=next_z,
            reward=self.reward_values[reward_logits.argmax(-1)],
            reward_mean=(reward_logits.softmax(-1) * self.reward_values).sum(-1),
            next_terminal=terminal, next_legal=legal)

    def latent_step(self, z, actions):
        result = self.step_details(z, actions)
        # Standard expected reward for value learning, categorical readout for
        # the strict decoded-game audit. Both are recorded and distinguished.
        return result['next_z'], result['reward_mean'], result['next_terminal'], result['next_legal'], result['event']
