"""Complete on-policy imagined games. Rules/decoder never advance a trajectory."""
import numpy as np
import torch

from common.models import REWARD_SCALE, masked_categorical, preprocess_observation
from common.parallel import model_snapshot, worker_model
from common.rollout import Rollout, normalized
from common.targets import td_lambda_targets
from algorithms.ppo_afterstate import decision_values


def initial_boards(seeds):
    # Only reset is allowed: obtain the same initial-state distribution as PPO.
    from gym2048_env import Gym2048Env
    boards = []
    for seed in seeds:
        env = Gym2048Env()
        try:
            board, _ = env.reset(seed=int(seed))
            boards.append(preprocess_observation(board))
        finally:
            env.close()
    return torch.stack(boards)


def draw(probabilities, rngs, active):
    values = probabilities.cpu().double().numpy()
    values /= values.sum(-1, keepdims=True)
    return torch.tensor([min(int(np.searchsorted(np.cumsum(p), rngs[i].random(), side='right')), len(p)-1)
                         for p, i in zip(values, active)], device=probabilities.device)


def pack_records(records, bootstraps, summaries, device, gamma, td_steps, td_lambda):
    output = [[] for _ in range(7)]
    for rows, bootstrap in zip(records, bootstraps):
        if not rows:
            continue
        for column in range(5):
            output[column].append(torch.stack([row[column] for row in rows]))
        values = torch.stack([row[5] for row in rows])
        state_values = torch.cat((values, bootstrap.reshape(1))).cpu().numpy()
        targets = td_lambda_targets([row[6] for row in rows], state_values, td_steps, gamma, td_lambda)
        returns = torch.as_tensor(targets, device=device)
        output[5].append(returns - values)
        output[6].append(returns)
    if not output[0]:
        raise RuntimeError('World predicts every initial state terminal; revalidate the world before training')
    return (*[torch.cat(parts) for parts in output], summaries)


@torch.no_grad()
def collect_games(model, episodes, gamma, seed, td_steps=10, td_lambda=.5, max_steps=10000):
    """Roll out F -> sampled learned chance -> G, without state re-encoding.

    Terminal is the learned terminal classifier OR no predicted legal actions.
    A time limit is a truncation with a critic bootstrap, never a terminal zero.
    Independent per-game action/chance RNGs preserve serial/worker ordering.
    """
    if episodes < 1 or max_steps < 1:
        raise ValueError('Episodes and max imagined steps must be positive')
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        z = model.world.encode(initial_boards(range(seed, seed + episodes)).to(device))
        records, summaries = [[] for _ in range(episodes)], [None] * episodes
        totals, bootstraps = [0.] * episodes, [None] * episodes
        zero_counts, zero_streaks, longest_zero = ([0] * episodes for _ in range(3))
        action_rng = [np.random.default_rng(seed + i + 20000000) for i in range(episodes)]
        chance_rng = [np.random.default_rng(seed + i + 30000000) for i in range(episodes)]
        terminal, legal = model.world.state_heads(z)
        alive = (terminal < 0) & (legal >= 0).any(-1)
        active = alive.nonzero().flatten().cpu().tolist()
        for i in range(episodes):
            if not bool(alive[i]):
                summaries[i] = dict(steps=0, spawn_return=0., terminated=True, truncated=False,
                                    zero_rewards=0, longest_zero_reward_streak=0)
        z = z[alive]
        masks = (legal >= 0)[alive]
        for step in range(max_steps):
            if not active:
                break
            candidates = model.candidate_latents(z)
            logits, afterstate_values = model.policy(candidates)
            distribution = masked_categorical(logits, masks)
            values = decision_values(distribution, afterstate_values)
            actions = draw(distribution.probs, action_rng, active)
            u = candidates[torch.arange(len(active), device=device), actions]
            probabilities = model.world.event_probabilities(z, u, actions)
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError('Nonfinite learned chance probabilities')
            events = draw(probabilities, chance_rng, active)
            nxt = model.world.event_step(u, events)
            reward_logits, done, legal = model.world.predict_outputs(u, events, nxt)
            if not all(torch.isfinite(t).all() for t in (nxt, reward_logits, done, legal)):
                raise FloatingPointError('Nonfinite frozen-world rollout')
            rewards = model.world.reward_values[reward_logits.argmax(-1)]
            next_masks = legal >= 0
            terminated = (done >= 0) | ~next_masks.any(-1)
            log_probs = distribution.log_prob(actions)
            truncation_values = None
            if step + 1 == max_steps and (~terminated).any():
                next_candidates = model.candidate_latents(nxt[~terminated])
                next_logits, next_values = model.policy(next_candidates)
                truncation_values = decision_values(masked_categorical(next_logits, next_masks[~terminated]), next_values)
            remaining = []
            tail_index = 0
            for j, i in enumerate(active):
                records[i].append((candidates[j], masks[j], actions[j], log_probs[j],
                                   distribution.logits[j], values[j], float(rewards[j]) / REWARD_SCALE))
                totals[i] += float(rewards[j])
                zero = float(rewards[j]) == 0.
                zero_counts[i] += int(zero)
                zero_streaks[i] = zero_streaks[i] + 1 if zero else 0
                longest_zero[i] = max(longest_zero[i], zero_streaks[i])
                ended = bool(terminated[j])
                truncated = step + 1 == max_steps and not ended
                if ended or truncated:
                    bootstraps[i] = torch.zeros_like(values[j]) if ended else truncation_values[tail_index]
                    if truncated:
                        tail_index += 1
                    summaries[i] = dict(steps=len(records[i]), spawn_return=totals[i],
                                        terminated=ended, truncated=truncated, zero_rewards=zero_counts[i],
                                        longest_zero_reward_streak=longest_zero[i])
                else:
                    remaining.append(i)
            if step + 1 == max_steps:
                break
            keep = ~terminated
            z, masks, active = nxt[keep], next_masks[keep], remaining
        return pack_records(records, bootstraps, summaries, device, gamma, td_steps, td_lambda)
    finally:
        model.train(was_training)


def _collect_worker(job):
    snapshot, episodes, gamma, seed, td_steps, td_lambda, max_steps = job
    return collect_games(worker_model(snapshot), episodes, gamma, seed, td_steps, td_lambda, max_steps)


def collect_imagined(model, episodes, gamma, seed, td_steps=10, td_lambda=.5,
                     max_steps=10000, pool=None, imagination_device='auto'):
    if imagination_device not in ('auto', 'model'):
        raise ValueError('imagination-device must be auto or model')
    if imagination_device == 'auto' and pool is not None and pool.workers > 1:
        snapshot = model_snapshot(model)
        groups = np.array_split(np.arange(episodes), min(pool.workers, episodes))
        parts = pool.map(_collect_worker, [(snapshot, len(ids), gamma, seed + int(ids[0]),
                                          td_steps, td_lambda, max_steps) for ids in groups])
        device = next(model.parameters()).device
        tensors = [torch.cat([part[i] for part in parts]).to(device) for i in range(7)]
        summaries = [episode for part in parts for episode in part[7]]
    else:
        *tensors, summaries = collect_games(model, episodes, gamma, seed, td_steps, td_lambda, max_steps)
    tensors[5] = normalized(tensors[5])
    return Rollout(*tensors, summaries)
