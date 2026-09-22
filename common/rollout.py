"""Collect complete games and construct finite TD(lambda) targets for A2C/PPO."""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
import torch
from common.models import REWARD_SCALE, preprocess_observation, masked_categorical
from common.targets import td_lambda_targets


def normalized(x):
    return (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-8)


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
def collect_actor_critic(env, model, episodes, gamma, seed, td_steps=10, td_lambda=.5, pool=None):
    """Batch active games; evaluate each visited state once and reuse critic predictions for TD(lambda).

    Each episode has its own action RNG. Worker completion order cannot change
    the trajectory order. Normalize advantages once across the entire update.
    """
    if episodes < 1:
        raise ValueError('Episodes must be positive')
    if pool is not None and pool.workers > 1:
        from common.parallel import model_snapshot
        snapshot = model_snapshot(model)
        groups = np.array_split(np.arange(episodes), min(pool.workers, episodes))
        jobs = [(snapshot, len(ids), gamma, seed + int(ids[0]), td_steps, td_lambda) for ids in groups]
        parts = pool.map(_collect_worker, jobs)
        device = next(model.parameters()).device
        tensors = [torch.cat([part[i] for part in parts]).to(device) for i in range(7)]
        summaries = [episode for part in parts for episode in part[7]]
    else:
        *tensors, summaries = _collect_games(env, model, episodes, gamma, seed, td_steps, td_lambda)
    tensors[5] = normalized(tensors[5])
    return Rollout(*tensors, summaries)


def _collect_worker(job):
    from gym2048_env import Gym2048Env
    from common.parallel import worker_model
    snapshot, episodes, gamma, seed, td_steps, td_lambda = job
    return _collect_games(Gym2048Env(), worker_model(snapshot), episodes, gamma, seed, td_steps, td_lambda)


@torch.no_grad()
def _collect_games(env, model, episodes, gamma, seed, td_steps, td_lambda):
    device = next(model.parameters()).device
    envs = [env] + [deepcopy(env) for _ in range(episodes - 1)]
    observations, infos, records = [], [], [[] for _ in envs]
    rngs = [np.random.default_rng(seed + i + 20_000_000) for i in range(episodes)]
    summaries, bootstraps = [None] * episodes, [None] * episodes
    totals = [0.] * episodes
    active = list(range(episodes))
    try:
        for i, game in enumerate(envs):
            state, info = game.reset(seed=seed + i)
            observations.append(state)
            infos.append(info)
        while active:
            states = torch.stack([preprocess_observation(observations[i]) for i in active]).to(device)
            logits, values = model(states)
            masks = torch.tensor([infos[i]['can_move_dir'] for i in active], device=device)
            dist = masked_categorical(logits, masks)
            probabilities = dist.probs.cpu().numpy().astype(np.float64)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            # Inverse CDF sampling, one independent draw per game and move.
            actions = torch.tensor([min(int(np.searchsorted(np.cumsum(probabilities[j]),
                                      rngs[i].random(), side='right')), 3)
                                    for j, i in enumerate(active)], device=device)
            logps = dist.log_prob(actions)
            remaining = []
            for j, i in enumerate(active):
                state, reward, terminated, truncated, info = envs[i].step(int(actions[j]))
                records[i].append((states[j], masks[j], actions[j], logps[j], dist.logits[j],
                                   values[j], float(reward) / REWARD_SCALE, terminated))
                totals[i] += float(reward)
                observations[i], infos[i] = state, info
                if terminated or truncated:
                    bootstraps[i] = (torch.zeros_like(values[j]) if terminated else
                                     model(preprocess_observation(state).to(device))[1])
                    summaries[i] = dict(info, steps=len(records[i]), spawn_return=totals[i])
                else:
                    remaining.append(i)
            active = remaining
    finally:
        for game in envs[1:]:
            if hasattr(game, 'close'):
                game.close()
    output = [[] for _ in range(7)]
    for rows, bootstrap in zip(records, bootstraps):
        for col in range(5):
            output[col].append(torch.stack([row[col] for row in rows]))
        values = torch.stack([row[5] for row in rows])
        state_values = torch.cat((values, bootstrap.reshape(1))).cpu().numpy()
        targets = td_lambda_targets([row[6] for row in rows], state_values, td_steps, gamma, td_lambda)
        returns = torch.as_tensor(targets, device=device)
        advantages = returns - values
        output[5].append(advantages)
        output[6].append(returns)
    return (*[torch.cat(parts) for parts in output], summaries)
