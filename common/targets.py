"""Finite-horizon TD(lambda) targets shared by policy and search learners."""
import numpy as np


def td_lambda_targets(rewards, values, td_steps=10, gamma=.999, td_lambda=.5):
    """Mix 1..n-step returns within one episode, retaining the final tail weight.

    rewards has T entries; values has T+1 frozen critic/search estimates. The
    last value is zero at true termination, or a bootstrap at a time limit.
    Arrays must never join different episodes. Rewards/values use the same units.
    """
    rewards, values = np.asarray(rewards, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if rewards.ndim != 1 or values.shape != (len(rewards) + 1,):
        raise ValueError('Expected T rewards and T+1 state values')
    if td_steps < 1 or not 0 <= gamma <= 1 or not 0 <= td_lambda <= 1:
        raise ValueError('td_steps must be positive; gamma and td_lambda must be in [0, 1]')
    targets = values[:-1].copy()
    for _ in range(min(td_steps, len(rewards))):
        continuation = np.concatenate((targets[1:], values[-1:]))
        targets = rewards + gamma * ((1 - td_lambda) * values[1:] + td_lambda * continuation)
    return targets.astype(np.float32)
