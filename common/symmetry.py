"""D4 board/action permutations: optional left-right mirror, then CCW rotation.

These transform coordinates, not tile values. Actions are LEFT, UP, RIGHT, DOWN.
All helpers accept a single transform or one transform per batch row.
"""
from functools import lru_cache

import torch


NAMES = ('identity', 'rot90', 'rot180', 'rot270',
         'mirror', 'mirror_rot90', 'mirror_rot180', 'mirror_rot270')


@lru_cache(maxsize=8)
def _permutations(device):
    cells, actions = [], []
    for mirror in (False, True):
        board = torch.arange(16).reshape(4, 4)
        policy = torch.arange(4)
        if mirror:
            board = board.flip(-1)
            policy = policy[[2, 1, 0, 3]]
        for rotation in range(4):
            cells.append(torch.rot90(board, rotation, (-2, -1)).reshape(16))
            actions.append(policy.roll(-rotation))
    cells = torch.stack(cells).to(device)
    actions = torch.stack(actions).to(device)
    return cells, actions, actions.argsort(-1)


def transform_boards(states, transforms):
    """Transform raw boards or exponents with shape [..., 16]."""
    if states.shape[-1] != 16:
        raise ValueError('Expected flattened 4x4 boards with final dimension 16')
    cells, _, _ = _permutations(str(states.device))
    return states.gather(-1, cells[transforms].expand_as(states))


def transform_policy(policy, transforms, *, inverse=False):
    """Permute probabilities, logits, or legal masks; inverse aligns to the input."""
    if policy.shape[-1] != 4:
        raise ValueError('Expected four actions in LEFT, UP, RIGHT, DOWN order')
    _, forward, backward = _permutations(str(policy.device))
    permutation = backward if inverse else forward
    return policy.gather(-1, permutation[transforms].expand_as(policy))


def transform_actions(actions, transforms, *, inverse=False):
    """Map original action indices into the transformed board's coordinates."""
    _, backward, forward = _permutations(str(actions.device))
    mapping = backward if inverse else forward
    return mapping[transforms].expand(*actions.shape, 4).gather(-1, actions[..., None]).squeeze(-1)
