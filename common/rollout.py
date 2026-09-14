"""Collect complete games and construct GAE targets for A2C/PPO."""
from dataclasses import dataclass
import numpy as np
import torch
from common.models import REWARD_SCALE, preprocess_observation, masked_categorical


def normalized(x):
    return (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-8)


def compute_gae(rewards, values, next_values, terminated, boundaries,
                gamma=1.0, gae_lambda=0.95):
    """Bootstrap time limits but never leak GAE across episode boundaries."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros((), device=rewards.device)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        carry = delta + gamma * gae_lambda * (~boundaries[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values


def discounted_returns(rewards, gamma):
    result = np.zeros(len(rewards), dtype=np.float32)
    carry = 0.0
    for i in reversed(range(len(rewards))):
        carry = float(rewards[i]) + gamma * carry
        result[i] = carry
    return result


@dataclass
class Rollout:
    states: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_logits: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    episodes: list


@torch.no_grad()
def collect_actor_critic(env, model, episodes, gamma, seed, gae_lambda=0.95):
    device = next(model.parameters()).device
    states, masks, actions, logps, old_logits = [], [], [], [], []
    rewards, values, next_values, terminateds, boundaries, summaries = [], [], [], [], [], []
    for ep in range(episodes):
        state, info = env.reset(seed=seed + ep)
        steps = 0
        spawn_return = 0.0
        while True:
            x = preprocess_observation(state).to(device)
            logits, value = model(x)
            mask = torch.tensor(info['can_move_dir'], device=device)
            dist = masked_categorical(logits, mask)
            action = dist.sample()
            state, reward, terminated, truncated, info = env.step(int(action))
            states.append(x); masks.append(mask); actions.append(action)
            logps.append(dist.log_prob(action)); old_logits.append(dist.logits)
            values.append(value)
            next_values.append(model(preprocess_observation(state).to(device))[1]
                               if not terminated else torch.zeros_like(value))
            rewards.append(float(reward) / REWARD_SCALE)
            spawn_return += float(reward)
            terminateds.append(terminated); boundaries.append(terminated or truncated)
            steps += 1
            if terminated or truncated:
                summaries.append(dict(info, steps=steps, spawn_return=spawn_return))
                break
    values = torch.stack(values)
    advantages, returns = compute_gae(
        torch.tensor(rewards, device=device), values, torch.stack(next_values),
        torch.tensor(terminateds, device=device), torch.tensor(boundaries, device=device),
        gamma, gae_lambda)
    return Rollout(torch.stack(states), torch.stack(masks), torch.stack(actions),
                   torch.stack(logps), torch.stack(old_logits), normalized(advantages),
                   returns, summaries)
