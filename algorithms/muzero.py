"""MuZero for 2048: representation h, deterministic dynamics g, prediction f.

Search uses learned latent transitions only. Real games provide search policies,
observed rewards and return targets for recurrent training. No board simulator,
chance nodes or afterstates are used inside the search tree.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gym2048_env import Gym2048Env
from common.models import BoardEncoder, REWARD_SCALE, preprocess_observation, masked_categorical
from common.checkpoints import read_checkpoint
from common.evaluation import evaluate, summarize_results
from common.parallel import model_snapshot, worker_model
from common.training import TrainingRun, training_parser, resolve_args


class MuZeroNetwork(nn.Module):
    """Small h/g/f model. Values are spawn/128; the reward head predicts raw spawn."""
    def __init__(self, obs_dim=16, num_actions=4, architecture='cnn2x2', latent_dim=128):
        super().__init__()
        if obs_dim != 16 or num_actions != 4 or latent_dim < 1:
            raise ValueError('MuZero requires a 4x4 board, four actions and a positive latent dimension')
        self.architecture, self.num_actions = architecture, num_actions
        self.encoder = BoardEncoder(obs_dim, num_actions, architecture)
        self.model_config = dict(self.encoder.model_config, model_type='muzero', latent_dim=latent_dim)
        self.representation = nn.Sequential(nn.Linear(256, latent_dim),
                                            nn.LayerNorm(latent_dim), nn.Tanh())
        self.dynamics = nn.Sequential(nn.Linear(latent_dim + num_actions, latent_dim), nn.ReLU())
        self.next_latent = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                        nn.LayerNorm(latent_dim), nn.Tanh())
        self.reward_head = nn.Linear(latent_dim, 1)
        self.policy_head = nn.Linear(latent_dim, num_actions)
        self.value_head = nn.Linear(latent_dim, 1)
        nn.init.orthogonal_(self.policy_head.weight, gain=.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        nn.init.zeros_(self.reward_head.weight)
        nn.init.constant_(self.reward_head.bias, 2.2)

    def h(self, observations):
        return self.representation(self.encoder.features(observations))

    def g(self, hidden, actions):
        actions = torch.as_tensor(actions, dtype=torch.long, device=hidden.device)
        one_hot = F.one_hot(actions, self.num_actions).to(hidden.dtype)
        features = self.dynamics(torch.cat((hidden, one_hot), dim=-1))
        return self.next_latent(features), self.reward_head(features).squeeze(-1)

    def f(self, hidden):
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


@dataclass
class Node:
    prior: float = 1.0
    hidden: torch.Tensor | None = None
    reward: float = 0.0  # Incoming edge reward, already divided by REWARD_SCALE.
    predicted_value: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0  # Return from this node, excluding its incoming reward.
    children: dict = field(default_factory=dict)

    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else self.predicted_value


def visit_policy(root, temperature=1.0):
    """Soft search target at tau=1; greedy visit-count decision at tau=0."""
    policy = np.zeros(4, dtype=np.float64)
    actions = list(root.children)
    if not actions:
        raise ValueError('Cannot select an action from a terminal root')
    counts = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)
    if not counts.any():
        counts = np.array([root.children[a].prior for a in actions])
    if temperature <= 0:
        policy[actions[int(counts.argmax())]] = 1.
    else:
        weights = np.full_like(counts, -np.inf)
        positive = counts > 0
        weights[positive] = np.log(counts[positive]) / temperature
        weights = np.exp(weights - weights.max())
        policy[actions] = weights / weights.sum()
    return policy


class MCTS:
    """One deterministic latent child per action; h once, g/f once per expansion."""
    def __init__(self, model, device, gamma=.999, search_depth=10, seed=0,
                 dirichlet_frac=.25, c_puct=1.25):
        if search_depth < 1 or not 0 <= gamma <= 1:
            raise ValueError('Search depth must be positive; gamma must be in [0, 1]')
        self.model, self.device = model, device
        self.gamma, self.search_depth = gamma, search_depth
        self.dirichlet_frac, self.c_puct = dirichlet_frac, c_puct
        self.rng = np.random.default_rng(seed)

    def expand(self, node, hidden, reward=0., legal_mask=None):
        logits, value = self.model.f(hidden)
        # Only the real root has a known legality mask; internal nodes use all actions.
        mask = np.ones(4, dtype=bool) if legal_mask is None else np.asarray(legal_mask, dtype=bool)
        priors = masked_categorical(logits, mask).probs.cpu().numpy()
        node.hidden, node.reward, node.predicted_value = hidden, reward, float(value)
        node.children = {int(a): Node(prior=float(priors[a])) for a in np.flatnonzero(mask)}

    def select(self, node, low, high):
        scale = max(high - low, 1.0)
        exploration = self.c_puct + math.log((node.visit_count + 19653) / 19652)
        def score(action):
            child = node.children[action]
            q = ((child.reward + self.gamma * child.value - low) / scale
                 if child.visit_count else 0.)
            u = exploration * child.prior * math.sqrt(node.visit_count + 1) / (child.visit_count + 1)
            return q + u
        return max(node.children, key=score)

    @torch.no_grad()
    def search(self, observation, legal_mask, simulations):
        if simulations < 1:
            raise ValueError('Search simulations must be positive')
        if not np.any(legal_mask):
            raise ValueError('Cannot search a terminal root')
        was_training = self.model.training
        self.model.eval()
        try:
            root = Node()
            self.expand(root, self.model.h(preprocess_observation(observation).to(self.device)),
                        legal_mask=legal_mask)
            if self.dirichlet_frac:
                noise = self.rng.dirichlet([.3] * len(root.children))
                for child, epsilon in zip(root.children.values(), noise):
                    child.prior = (1 - self.dirichlet_frac) * child.prior + self.dirichlet_frac * epsilon
            low, high = float('inf'), -float('inf')
            for _ in range(simulations):
                node, path = root, [root]
                for _depth in range(self.search_depth):
                    action = self.select(node, low, high)
                    child = node.children[action]
                    path.append(child)
                    if child.hidden is None:
                        hidden, reward = self.model.g(node.hidden, action)
                        self.expand(child, hidden, float(reward) / REWARD_SCALE)
                        node = child
                        break
                    node = child
                # A depth limit is a bootstrap, never an artificial terminal state.
                value = node.predicted_value
                for visited in reversed(path):
                    visited.visit_count += 1
                    visited.value_sum += value
                    if visited is not root:
                        q = visited.reward + self.gamma * visited.value
                        low, high = min(low, q), max(high, q)
                    value = visited.reward + self.gamma * value
            return root
        finally:
            self.model.train(was_training)


@dataclass
class GameHistory:
    observations: np.ndarray  # T+1 actual boards, preprocessed to tile exponents.
    actions: np.ndarray       # T actions; histories always end at true termination.
    rewards: np.ndarray       # T observed rewards, in raw spawn units (2 or 4).
    policies: np.ndarray      # T soft root-visit targets, including greedy behavior steps.
    root_values: np.ndarray   # T search values, in spawn/128 units.

    def __len__(self):
        return len(self.actions)

    def target_value(self, index, td_steps=10, gamma=.999, td_lambda=.5):
        """Truncated lambda-return using real spawn rewards and stored search values.

        Mix 1..n-step returns, retaining the remaining weight at the horizon.
        Terminal bootstrap is zero.
        """
        if index >= len(self):
            return 0.
        end = min(index + td_steps, len(self))
        value = float(self.root_values[end]) if end < len(self) else 0.
        for k in range(end - 1, index - 1, -1):
            next_value = float(self.root_values[k + 1]) if k + 1 < len(self) else 0.
            value = float(self.rewards[k]) / REWARD_SCALE + gamma * (
                (1 - td_lambda) * next_value + td_lambda * value)
        return value


def self_play_game(model, device, mcts_sims, temperature_moves, seed, gamma=.999, search_depth=10):
    planner = MCTS(model, device, gamma, search_depth, seed + 10_000_000)
    behavior_rng = np.random.default_rng(seed + 20_000_000)
    env = Gym2048Env()
    state, info = env.reset(seed=seed)
    observations = [preprocess_observation(state).numpy()]
    actions, rewards, policies, values = [], [], [], []
    try:
        while True:
            root = planner.search(state, info['can_move_dir'], mcts_sims)
            target = visit_policy(root)
            behavior = visit_policy(root, 1. if len(actions) < temperature_moves else 0.)
            action = int(behavior_rng.choice(4, p=behavior))
            state, reward, done, truncated, info = env.step(action)
            observations.append(preprocess_observation(state).numpy())
            actions.append(action)
            rewards.append(reward)
            policies.append(target)
            values.append(root.value)
            if truncated:
                raise ValueError('MuZero replay requires complete episodes, not time-limit truncation')
            if done:
                break
    finally:
        env.close()
    history = GameHistory(np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.int64),
                          np.asarray(rewards, dtype=np.float32), np.asarray(policies, dtype=np.float32),
                          np.asarray(values, dtype=np.float32))
    return history, dict(info, steps=len(history), spawn_return=float(sum(rewards)))


@dataclass
class SequenceBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    policies: torch.Tensor
    values: torch.Tensor
    policy_mask: torch.Tensor


def make_batch(samples, unroll_steps, td_steps, gamma, device, td_lambda=.5):
    """Actual actions up to termination, then absorbing actions/rewards/values.

    Each sample is (complete_game, start_index). Only its initial board is input;
    future boards never replace recurrent hidden states during optimization.
    """
    batch_size = len(samples)
    observations = np.stack([game.observations[start] for game, start in samples])
    actions = np.zeros((batch_size, unroll_steps), dtype=np.int64)
    rewards = np.zeros((batch_size, unroll_steps), dtype=np.float32)
    policies = np.zeros((batch_size, unroll_steps + 1, 4), dtype=np.float32)
    values = np.zeros((batch_size, unroll_steps + 1), dtype=np.float32)
    policy_mask = np.zeros((batch_size, unroll_steps + 1), dtype=bool)
    for row, (game, start) in enumerate(samples):
        if not 0 <= start < len(game):
            raise ValueError('A replay sample must start at a real decision state')
        for k in range(unroll_steps + 1):
            index = start + k
            values[row, k] = game.target_value(index, td_steps, gamma, td_lambda)
            if index < len(game):
                policies[row, k] = game.policies[index]
                policy_mask[row, k] = True
            if k < unroll_steps:
                if index < len(game):
                    actions[row, k], rewards[row, k] = game.actions[index], game.rewards[index]
                else:
                    # Teach that every action after termination has zero future reward.
                    actions[row, k] = np.random.randint(4)
    return SequenceBatch(*(torch.as_tensor(array, device=device) for array in
                           (observations, actions, rewards, policies, values, policy_mask)))


def sample_batch(histories, batch_size, unroll_steps, td_steps, gamma, device, td_lambda=.5):
    """Uniform over recorded transitions, with replacement; never cross games."""
    cumulative = np.cumsum([len(game) for game in histories])
    if not len(cumulative) or cumulative[-1] == 0:
        raise ValueError('Cannot sample an empty replay buffer')
    indices = np.random.randint(int(cumulative[-1]), size=batch_size)
    games = np.searchsorted(cumulative, indices, side='right')
    starts = indices - np.concatenate(([0], cumulative[:-1]))[games]
    return make_batch([(histories[g], int(start)) for g, start in zip(games, starts)],
                      unroll_steps, td_steps, gamma, device, td_lambda)


def train_batch(model, optimizer, batch, value_coef=.5, reward_coef=1.):
    """Backpropagate policy/value/reward losses through h and the entire g unroll."""
    model.train()
    hidden = model.h(batch.observations)
    logits, values = model.f(hidden)
    all_logits, all_values, all_rewards = [logits], [values], []
    for k in range(batch.actions.shape[1]):
        hidden, rewards = model.g(hidden, batch.actions[:, k])
        logits, values = model.f(hidden)
        all_logits.append(logits)
        all_values.append(values)
        all_rewards.append(rewards)
    logits, values = torch.stack(all_logits, dim=1), torch.stack(all_values, dim=1)
    rewards = torch.stack(all_rewards, dim=1)
    # Unmasked CE teaches zero-target illegal actions to lose probability. There
    # is no policy target after termination, but value/reward targets stay zero.
    ce = -(batch.policies * F.log_softmax(logits, dim=-1)).sum(-1)
    policy_loss = (ce * batch.policy_mask).sum() / batch.policy_mask.sum().clamp_min(1)
    value_loss = F.mse_loss(values, batch.values)
    reward_loss = F.mse_loss(rewards, batch.rewards)
    loss = policy_loss + value_coef * value_loss + reward_coef * reward_loss
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return dict(loss=float(loss.detach()), policy_loss=float(policy_loss.detach()),
                value_loss=float(value_loss.detach()), reward_loss=float(reward_loss.detach()),
                grad_norm=float(grad_norm))


def _self_play_worker(job):
    snapshot, sims, temperature, seed, gamma, depth = job
    return self_play_game(worker_model(snapshot), 'cpu', sims, temperature, seed, gamma, depth)


def collect_self_play(model, device, games, sims, temperature, seed, gamma, depth, pool):
    if pool.workers > 1:
        snapshot = model_snapshot(model)
        return pool.map(_self_play_worker, [(snapshot, sims, temperature, seed + i, gamma, depth)
                                           for i in range(games)])
    return [self_play_game(model, device, sims, temperature, seed + i, gamma, depth)
            for i in range(games)]


def search_controller(model, simulations=100, gamma=.999, search_depth=10, seed=0):
    planner = MCTS(model, next(model.parameters()).device, gamma, search_depth, seed, dirichlet_frac=0.)
    def action(state, info):
        root = planner.search(state, info['can_move_dir'], simulations)
        return int(visit_policy(root, temperature=0.).argmax())
    return action


def _search_episode(model, seed, sims, gamma, depth):
    action = search_controller(model, sims, gamma, depth, seed)
    return evaluate(model, 1, seed, lambda state, info, _: action(state, info))['results'][0]


def _search_worker(job):
    snapshot, seed, sims, gamma, depth = job
    return _search_episode(worker_model(snapshot), seed, sims, gamma, depth)


def evaluate_search(model, episodes=10, seed=1_000_000, mcts_sims=100, gamma=.999, search_depth=10, pool=None):
    if episodes < 1:
        raise ValueError('Evaluation requires at least one episode')
    started = time.perf_counter()
    if pool is not None and pool.workers > 1:
        snapshot = model_snapshot(model)
        results = pool.map(_search_worker, [(snapshot, seed + i, mcts_sims, gamma, search_depth)
                                           for i in range(episodes)])
    else:
        results = [_search_episode(model, seed + i, mcts_sims, gamma, search_depth) for i in range(episodes)]
    return summarize_results(results, time.perf_counter() - started)


def train(args):
    factory = lambda: MuZeroNetwork(architecture=args.architecture, latent_dim=args.latent_dim)
    with TrainingRun(args, 'muzero', model_factory=factory) as run:
        model, optimizer, device = run.model, run.optimizer, run.device
        replay = deque(maxlen=args.buffer_games)
        if run.saved:
            if 'replay' not in run.saved:
                raise ValueError('MuZero resume requires a full checkpoint with replay histories')
            replay.extend(GameHistory(**game) for game in run.saved['replay'])
        def validation():
            return evaluate_search(model, args.eval_episodes, args.eval_seed, args.eval_mcts_sims,
                                   args.gamma, args.search_depth, run.pool)
        run.ensure_baseline(validation, {'replay': [vars(game) for game in replay]})
        for iteration in range(run.start, args.iterations + 1):
            # 1. Frozen network generates complete real episodes using latent MCTS.
            started = time.perf_counter()
            games = collect_self_play(model, device, args.self_play_games, args.mcts_sims,
                                      args.temperature_moves, 10_000_000 + args.seed + iteration * 100_000,
                                      args.gamma, args.search_depth, run.pool)
            summaries = []
            for game_index, (history, summary) in enumerate(games):
                replay.append(history)
                summaries.append(summary)
                print(json.dumps(dict(iteration=iteration, game=game_index + 1,
                                      steps=len(history), max_tile=summary['max_value'])), flush=True)
            collect_seconds = time.perf_counter() - started
            # 2. Sample sequences, unroll learned dynamics, jointly train h/g/f.
            started = time.perf_counter()
            histories = list(replay)
            losses = [train_batch(model, optimizer,
                        sample_batch(histories, args.batch_size, args.unroll_steps, args.td_steps,
                                     args.gamma, device, args.td_lambda), args.value_coef, args.reward_coef)
                      for _ in range(args.train_steps)]
            metrics = {key: float(np.mean([loss[key] for loss in losses])) for key in losses[0]}
            metrics.update(iteration=iteration, updates=args.train_steps, replay_games=len(replay),
                           gamma=args.gamma, td_steps=args.td_steps, td_lambda=args.td_lambda,
                           replay_size=sum(map(len, replay)), transitions=sum(s['steps'] for s in summaries),
                           collect_seconds=collect_seconds, update_seconds=time.perf_counter() - started,
                           train_mean_return=float(np.mean([s['spawn_return'] for s in summaries])),
                           train_mean_steps=float(np.mean([s['steps'] for s in summaries])),
                           train_max_tile=max(s['max_value'] for s in summaries))
            # 3. Noise-free validation, resumable replay checkpoint, periodic plot.
            if run.should_validate(iteration):
                metrics['validation'] = validation()
            run.record(metrics, {'replay': [vars(game) for game in replay]})
        return model


def main(argv=None):
    parser = training_parser(__doc__)
    for flag in ('self-play-games', 'mcts-sims', 'temperature-moves', 'buffer-games', 'batch-size',
                 'train-steps', 'eval-mcts-sims', 'latent-dim', 'unroll-steps', 'search-depth'):
        parser.add_argument('--' + flag, type=int)
    parser.add_argument('--td-steps', type=int, help='Positive TD(lambda) horizon (default 10); inherited on resume')
    parser.add_argument('--td-lambda', type=float, help='Truncated lambda-return mixing factor (default 0.5)')
    parser.add_argument('--value-coef', type=float)
    parser.add_argument('--reward-coef', type=float)
    requested = parser.parse_args(argv)
    saved = read_checkpoint(requested.resume) if requested.resume else None
    args = resolve_args(parser, 'muzero', dict(self_play_games=8, mcts_sims=100, temperature_moves=30,
        buffer_games=128, batch_size=64, train_steps=100, eval_every=10, plot_every=10, eval_mcts_sims=100,
        latent_dim=128, unroll_steps=5, gamma=.999, td_steps=10, td_lambda=.5,
        search_depth=10, value_coef=.5, reward_coef=1.), argv)
    for key in ('buffer_games', 'latent_dim', 'unroll_steps', 'search_depth'):
        if getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    if args.temperature_moves < 0 or args.value_coef <= 0 or args.reward_coef <= 0:
        parser.error('temperature-moves must be nonnegative; loss coefficients must be positive')
    if saved:
        if 'optimizer' not in saved or 'replay' not in saved:
            raise ValueError('MuZero resume requires a full training checkpoint with optimizer and replay')
        if args.latent_dim != saved['model_config']['latent_dim']:
            parser.error('Resume must preserve latent-dim')
        changed = args.search_depth != saved['config']['search_depth']
        new_directory = args.save_dir and Path(args.save_dir).resolve() != Path(args.resume).parent.resolve()
        if changed and not new_directory:
            parser.error('Changed search depth requires a new --save-dir to re-evaluate the baseline')
    return train(args)


if __name__ == '__main__':
    main()
