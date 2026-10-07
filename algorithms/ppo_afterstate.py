"""PPO over deterministic afterstates, followed by a separate random tile spawn."""
from copy import deepcopy
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gym2048_afterstate_env import Gym2048AfterstateEnv, candidate_afterstates
from common.models import BoardEncoder, REWARD_SCALE, masked_categorical
from common.parallel import model_snapshot, worker_model
from common.rollout import Rollout, normalized
from common.targets import td_lambda_targets
from common.evaluation import summarize_results
from common.training import TrainingRun, training_parser, resolve_args


# Best recorded PPO run: ppo_value_targets/spawn_td_seed0, iteration 15000,
# mean validation spawn return 12711.6 (also pretrained/ppo_cnn2x2.pt).
DEFAULTS = dict(architecture='cnn2x2', episodes_per_update=8, epochs=4, batch_size=256,
                gamma=.999, lr=3e-4, td_steps=10, td_lambda=.5, entropy_coef=.01,
                value_coef=.5, clip_range=.2, target_kl=.02)
KL_METHOD = 'masked_log_prob_v1'


class AfterstateActorCritic(BoardEncoder):
    """A shared encoder scores each candidate board before the random tile.

    pi(a|s) = masked_softmax(score(move(s,a))). W(x) estimates
    E[spawn_reward/128 + gamma * V(next_state) | x], and the decision-state
    baseline is V(s) = sum_a pi(a|s) * W(move(s,a)).
    """
    def __init__(self, obs_dim=16, num_actions=4, architecture='cnn2x2', afterstate_version=1):
        if num_actions != 4 or afterstate_version != 1:
            raise ValueError('PPO afterstate requires four actions and afterstate_version=1')
        super().__init__(obs_dim, num_actions, architecture)
        self.model_config.update(model_type='ppo_afterstate', afterstate_version=afterstate_version)
        self.policy_head = nn.Linear(256, 1)
        self.value_head = nn.Linear(256, 1)
        nn.init.orthogonal_(self.policy_head.weight, gain=.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, afterstates):
        # Single decision: [4,16]; batch: [B,4,16], all in tile exponents.
        if afterstates.ndim not in (2, 3) or afterstates.shape[-2:] != (4, 16):
            raise ValueError('Expected candidate afterstates with shape [4,16] or [B,4,16]')
        features = self.features(afterstates.reshape(-1, 16))
        shape = afterstates.shape[:-1]
        return (self.policy_head(features).reshape(shape),
                self.value_head(features).reshape(shape))


def encode_candidates(observations, device='cpu'):
    """Pure move enumeration and batched encoding; no chance sampling."""
    candidates, masks, _ = zip(*[candidate_afterstates(state) for state in observations])
    boards = np.stack(candidates).reshape(len(observations), 4, 16).astype(np.float32)
    exponents = np.zeros_like(boards)
    np.log2(boards, out=exponents, where=boards > 0)
    return torch.from_numpy(exponents).to(device), torch.as_tensor(np.stack(masks), device=device)


def decision_values(distribution, afterstate_values):
    """Detach policy weights: the critic must not optimize action probabilities."""
    return (distribution.probs.detach() * afterstate_values).sum(-1)


@torch.no_grad()
def afterstate_action(model, state, info=None):
    device = next(model.parameters()).device
    candidates, masks = encode_candidates([state], device)
    logits, _ = model(candidates)
    return int(masked_categorical(logits, masks).probs.argmax(-1).item())


@torch.no_grad()
def collect_afterstate(env, model, episodes, gamma, seed, td_steps=10, td_lambda=.5, pool=None):
    if episodes < 1:
        raise ValueError('Episodes must be positive')
    if pool is not None and pool.workers > 1:
        snapshot = model_snapshot(model)
        groups = np.array_split(np.arange(episodes), min(pool.workers, episodes))
        parts = pool.map(_collect_worker, [(snapshot, len(ids), gamma, seed + int(ids[0]),
                                          td_steps, td_lambda) for ids in groups])
        device = next(model.parameters()).device
        tensors = [torch.cat([part[i] for part in parts]).to(device) for i in range(7)]
        summaries = [episode for part in parts for episode in part[7]]
    else:
        *tensors, summaries = _collect_games(env, model, episodes, gamma, seed, td_steps, td_lambda)
    tensors[5] = normalized(tensors[5])
    # Rollout.states holds ALL four afterstates so PPO can reproduce its policy.
    return Rollout(*tensors, summaries)


def _collect_worker(job):
    snapshot, episodes, gamma, seed, td_steps, td_lambda = job
    return _collect_games(Gym2048AfterstateEnv(), worker_model(snapshot), episodes,
                          gamma, seed, td_steps, td_lambda)


@torch.no_grad()
def _collect_games(env, model, episodes, gamma, seed, td_steps, td_lambda):
    device = next(model.parameters()).device
    envs = [env] + [deepcopy(env) for _ in range(episodes - 1)]
    observations = [game.reset(seed=seed + i)[0] for i, game in enumerate(envs)]
    records, summaries = [[] for _ in envs], [None] * episodes
    totals, bootstraps = [0.] * episodes, [None] * episodes
    rngs = [np.random.default_rng(seed + i + 20_000_000) for i in range(episodes)]
    active = list(range(episodes))
    try:
        while active:
            candidates, masks = encode_candidates([observations[i] for i in active], device)
            logits, afterstate_values = model(candidates)
            distribution = masked_categorical(logits, masks)
            values = decision_values(distribution, afterstate_values)
            probabilities = distribution.probs.cpu().numpy().astype(np.float64)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            actions = torch.tensor([min(int(np.searchsorted(np.cumsum(probabilities[j]),
                                      rngs[i].random(), side='right')), 3)
                                    for j, i in enumerate(active)], device=device)
            log_probs = distribution.log_prob(actions)
            remaining = []
            for j, i in enumerate(active):
                # Deterministic transition contributes no spawn-mass reward.
                envs[i].step_move(int(actions[j]))
                state, reward, terminated, truncated, info = envs[i].step_spawn()
                records[i].append((candidates[j], masks[j], actions[j], log_probs[j],
                                   distribution.logits[j], values[j], float(reward) / REWARD_SCALE))
                totals[i] += float(reward)
                observations[i] = state
                if terminated or truncated:
                    if terminated:
                        bootstraps[i] = torch.zeros_like(values[j])
                    else:
                        next_candidates, next_masks = encode_candidates([state], device)
                        next_logits, next_values = model(next_candidates)
                        bootstraps[i] = decision_values(masked_categorical(next_logits, next_masks), next_values)[0]
                    summaries[i] = dict(steps=len(records[i]), spawn_return=totals[i],
                                        max_value=info['max_value'], merge_score=info['merge_score'])
                else:
                    remaining.append(i)
            active = remaining
    finally:
        for game in envs[1:]:
            game.close()
    output = [[] for _ in range(7)]
    for rows, bootstrap in zip(records, bootstraps):
        for column in range(5):
            output[column].append(torch.stack([row[column] for row in rows]))
        values = torch.stack([row[5] for row in rows])
        state_values = torch.cat((values, bootstrap.reshape(1))).cpu().numpy()
        targets = td_lambda_targets([row[6] for row in rows], state_values, td_steps, gamma, td_lambda)
        returns = torch.as_tensor(targets, device=device)
        output[5].append(returns - values)
        output[6].append(returns)
    return (*[torch.cat(parts) for parts in output], summaries)


@torch.no_grad()
def masked_policy_kl(old_log_probs, new_log_probs, legal_mask):
    """KL(old || new) from normalized categorical log-probs on the SAME mask.

    A legal softmax probability can flush to zero on MPS even when its log-prob
    is finite. Do not interpret that rounded zero as a loss of policy support.
    Clear illegal entries BEFORE subtraction, so masked -inf/NaN cannot make
    0 * NaN contaminate the sum. No epsilon floor changes the legal probabilities.
    """
    mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=new_log_probs.device)
    old = old_log_probs.masked_fill(~mask, 0.)
    new = new_log_probs.masked_fill(~mask, 0.)
    return (old.exp() * (old - new)).masked_fill(~mask, 0.).sum(-1)


@torch.no_grad()
def _rollout_diagnostics(model, rollout, batch_size):
    """Evaluate the FINAL policy on all saved candidates without RNG/gradients."""
    kl_sum = entropy_sum = 0.
    span_max, min_log_prob = 0., float('inf')
    zero_probs = 0
    for start in range(0, len(rollout.actions), batch_size):
        rows = slice(start, start + batch_size)
        mask = rollout.masks[rows]
        logits, _ = model(rollout.states[rows])
        distribution = masked_categorical(logits, mask)
        old = torch.distributions.Categorical(logits=rollout.old_logits[rows])
        kl = masked_policy_kl(old.logits, distribution.logits, mask)
        if not torch.isfinite(kl).all():
            raise FloatingPointError('Nonfinite afterstate rollout KL; check legal log-probs and model weights')
        kl_sum += float(kl.sum())
        entropy_sum += float(distribution.entropy().sum())
        span = logits.masked_fill(~mask, -torch.inf).amax(-1) - logits.masked_fill(~mask, torch.inf).amin(-1)
        span_max = max(span_max, float(span.max()))
        min_log_prob = min(min_log_prob, float(distribution.logits.masked_fill(~mask, torch.inf).min()))
        zero_probs += int(((distribution.probs == 0) & mask).sum())
    return dict(rollout_kl=kl_sum / len(rollout.actions),
                rollout_entropy=entropy_sum / len(rollout.actions),
                legal_logit_span_max=span_max, min_legal_log_prob=min_log_prob,
                zero_legal_probabilities=zero_probs)


def update(model, optimizer, rollout, epochs=4, batch_size=256, clip_range=.2,
           entropy_coef=.01, value_coef=.5, target_kl=.02):
    transitions = len(rollout.actions)
    metrics = dict(actor=0., critic=0., entropy=0., kl=0., updates=0,
                   kl_method=KL_METHOD, kl_checks=0, kl_nonfinite_checks=0,
                   planned_updates=epochs * math.ceil(transitions / batch_size),
                   samples_updated=0, update_stop_reason='epochs_complete')
    loss_totals = dict(actor=0., critic=0., entropy=0.)
    stopped = False
    for _ in range(epochs):
        indices = torch.randperm(transitions, device=rollout.states.device)
        for ids in indices.split(batch_size):
            logits, afterstate_values = model(rollout.states[ids])
            distribution = masked_categorical(logits, rollout.masks[ids])
            old_distribution = torch.distributions.Categorical(logits=rollout.old_logits[ids])
            kl = masked_policy_kl(old_distribution.logits, distribution.logits, rollout.masks[ids]).mean()
            metrics['kl'] = float(kl)
            metrics['kl_checks'] += 1
            if not math.isfinite(metrics['kl']):
                raise FloatingPointError('Nonfinite afterstate minibatch KL; check legal log-probs and model weights')
            if target_kl > 0 and metrics['kl'] > target_kl:
                metrics['update_stop_reason'] = 'target_kl'
                stopped = True
                break
            ratio = (distribution.log_prob(rollout.actions[ids]) - rollout.old_log_probs[ids]).exp()
            advantages = rollout.advantages[ids]
            actor = -torch.minimum(ratio * advantages,
                                   ratio.clamp(1 - clip_range, 1 + clip_range) * advantages).mean()
            # The return begins with the spawn AFTER the selected afterstate.
            chosen_values = afterstate_values.gather(-1, rollout.actions[ids, None]).squeeze(-1)
            critic = F.smooth_l1_loss(chosen_values, rollout.returns[ids])
            entropy = distribution.entropy().mean()
            loss = actor + value_coef * critic - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            metrics.update(actor=float(actor.detach()), critic=float(critic.detach()),
                           entropy=float(entropy.detach()), updates=metrics['updates'] + 1)
            metrics['samples_updated'] += len(ids)
            for name in loss_totals:
                loss_totals[name] += metrics[name] * len(ids)
        if stopped:
            break
    metrics.update(effective_epochs=metrics['samples_updated'] / transitions,
                   update_fraction=metrics['updates'] / metrics['planned_updates'],
                   **{name + '_mean': total / max(1, metrics['samples_updated'])
                      for name, total in loss_totals.items()})
    started = time.perf_counter()
    metrics.update(_rollout_diagnostics(model, rollout, batch_size))
    metrics['diagnostic_seconds'] = time.perf_counter() - started
    return metrics


def evaluate_afterstate(model, episodes=10, seed=1_000_000, pool=None):
    if episodes < 1:
        raise ValueError('Evaluation requires at least one episode')
    started = time.perf_counter()
    if pool is not None and pool.workers > 1:
        snapshot = model_snapshot(model)
        groups = np.array_split(np.arange(episodes), min(pool.workers, episodes))
        parts = pool.map(_evaluate_worker, [(snapshot, len(ids), seed + int(ids[0])) for ids in groups])
        results = [row for part in parts for row in part]
    else:
        results = _evaluate_games(model, episodes, seed)
    return summarize_results(results, time.perf_counter() - started)


def _evaluate_worker(job):
    snapshot, episodes, seed = job
    return _evaluate_games(worker_model(snapshot), episodes, seed)


@torch.no_grad()
def _evaluate_games(model, episodes, seed):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    envs = [Gym2048AfterstateEnv() for _ in range(episodes)]
    observations = [env.reset(seed=seed + i)[0] for i, env in enumerate(envs)]
    totals, steps, results = [0.] * episodes, [0] * episodes, [None] * episodes
    active = list(range(episodes))
    try:
        while active:
            candidates, masks = encode_candidates([observations[i] for i in active], device)
            logits, _ = model(candidates)
            actions = masked_categorical(logits, masks).probs.argmax(-1).cpu().tolist()
            remaining = []
            for i, action in zip(active, actions):
                envs[i].step_move(action)
                state, reward, terminated, truncated, info = envs[i].step_spawn()
                observations[i] = state
                totals[i] += reward
                steps[i] += 1
                if terminated or truncated:
                    results[i] = dict(seed=seed + i, steps=steps[i], spawn_return=totals[i],
                                      board_sum=int(state.sum()), max_tile=info['max_value'],
                                      merge_score=info['merge_score'], legacy_score=info['score'])
                else:
                    remaining.append(i)
            active = remaining
    finally:
        model.train(was_training)
        for env in envs:
            env.close()
    return results


def train(args):
    with TrainingRun(args, 'ppo_afterstate',
                     model_factory=lambda: AfterstateActorCritic(architecture=args.architecture)) as run:
        env = Gym2048AfterstateEnv()
        try:
            run.ensure_baseline(lambda: evaluate_afterstate(run.model, args.eval_episodes, args.eval_seed, run.pool))
            for iteration in range(run.start, args.iterations + 1):
                started = time.perf_counter()
                rollout = collect_afterstate(env, run.model, args.episodes_per_update, args.gamma,
                                            10_000_000 + args.seed + iteration * 100_000,
                                            args.td_steps, args.td_lambda, run.pool)
                collect_seconds = time.perf_counter() - started
                started = time.perf_counter()
                metrics = update(run.model, run.optimizer, rollout, args.epochs, args.batch_size,
                                 args.clip_range, args.entropy_coef, args.value_coef, args.target_kl)
                metrics.update(collect_seconds=collect_seconds, update_seconds=time.perf_counter() - started,
                               gamma=args.gamma, td_steps=args.td_steps, td_lambda=args.td_lambda,
                               iteration=iteration, transitions=len(rollout.actions),
                               train_mean_return=float(np.mean([e['spawn_return'] for e in rollout.episodes])),
                               train_mean_steps=float(np.mean([e['steps'] for e in rollout.episodes])),
                               train_max_tile=max(e['max_value'] for e in rollout.episodes))
                if run.should_validate(iteration):
                    metrics['validation'] = evaluate_afterstate(run.model, args.eval_episodes, args.eval_seed, run.pool)
                run.record(metrics)
            return run.model
        finally:
            env.close()


def main(argv=None):
    parser = training_parser(__doc__)
    for flag in ('episodes-per-update', 'epochs', 'batch-size', 'td-steps'):
        parser.add_argument('--' + flag, type=int)
    for flag in ('td-lambda', 'entropy-coef', 'value-coef', 'clip-range', 'target-kl'):
        parser.add_argument('--' + flag, type=float)
    args = resolve_args(parser, 'ppo_afterstate', DEFAULTS, argv)
    # Metadata describes this implementation, including resumes of older runs.
    args.kl_method = KL_METHOD
    if not 0 < args.clip_range < 1 or args.entropy_coef < 0 or args.value_coef < 0 or args.target_kl < 0:
        parser.error('clip-range must be in (0, 1); loss coefficients and target-kl must be nonnegative')
    return train(args)


if __name__ == '__main__':
    main()
