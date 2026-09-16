"""Stochastic AlphaZero-style 2048: PUCT action edges with sampled chance outcomes.

Search backs up r + gamma*V, with the same spawn/128 return-to-go target used
in training. Search outcomes never consume the real environment's random stream.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from board import Board
from gym2048_env import Gym2048Env
from common.models import ActorCritic, REWARD_SCALE, preprocess_observation, masked_categorical
from common.rollout import discounted_returns
from common.evaluation import evaluate, summarize_results
from common.parallel import model_snapshot, worker_model
from common.training import TrainingRun, training_parser, resolve_args


obs_to_tensor = preprocess_observation


class PolicyValueNet(ActorCritic):
    """Unbounded value head for remaining spawned mass, no batch statistics."""


@dataclass
class Node:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict = field(default_factory=dict)  # decision -> action edge
    legal_mask: np.ndarray | None = None
    matrix: np.ndarray | None = None
    outcomes: dict = field(default_factory=dict)  # action edge -> chance states

    @property
    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class MCTS:
    def __init__(self, model, device, c_puct=1.5, dirichlet_alpha=0.3,
                 dirichlet_frac=0.25, gamma=1.0, seed=0):
        self.model, self.device = model, device
        self.c_puct, self.dirichlet_alpha = c_puct, dirichlet_alpha
        self.dirichlet_frac, self.gamma = dirichlet_frac, gamma
        self.rng = random.Random(seed)
        self.noise_rng = np.random.default_rng(seed)

    @torch.no_grad()
    def evaluate_and_expand(self, node):
        if node.matrix is None:
            raise ValueError('Decision node must carry a board')
        if node.legal_mask is None:
            node.legal_mask = np.asarray(Board(node.matrix).can_move_dir, dtype=bool)
        if not np.any(node.legal_mask):
            return 0.0
        logits, value = self.model(obs_to_tensor(node.matrix).to(self.device))
        probs = masked_categorical(logits, node.legal_mask).probs.cpu().numpy()
        for action in np.flatnonzero(node.legal_mask):
            node.children.setdefault(int(action), Node(prior=float(probs[action])))
        return float(value)

    @torch.no_grad()
    def run(self, root, mcts_sims):
        if mcts_sims < 1:
            raise ValueError('mcts_sims must be positive')
        was_training = self.model.training
        self.model.eval()
        try:
            if not root.children:
                self.evaluate_and_expand(root)
            if not root.children:
                return
            if self.dirichlet_frac > 0:
                noise = self.noise_rng.dirichlet([self.dirichlet_alpha] * len(root.children))
                for edge, epsilon in zip(root.children.values(), noise):
                    edge.prior = (1 - self.dirichlet_frac) * edge.prior + self.dirichlet_frac * float(epsilon)
            for _ in range(mcts_sims):
                node, path = root, []
                while node.children:
                    visited = [edge.q_value for edge in node.children.values() if edge.visit_count]
                    low, high = (min(visited), max(visited)) if visited else (0.0, 0.0)
                    scale = max(high - low, 1.0)
                    sqrt_visits = math.sqrt(1 + sum(edge.visit_count for edge in node.children.values()))
                    def puct(action):
                        edge = node.children[action]
                        q = (edge.q_value - low) / scale if edge.visit_count else 0.0
                        return q + self.c_puct * edge.prior * sqrt_visits / (1 + edge.visit_count)
                    action = max(node.children, key=puct)
                    edge = node.children[action]
                    # Resample the chance event on EVERY visit to this action.
                    board = Board(node.matrix, rng=self.rng)
                    board.step(action)
                    reward = board.reward / REWARD_SCALE
                    key = board.matrix.tobytes()
                    if key not in edge.outcomes:
                        edge.outcomes[key] = Node(1.0, matrix=board.matrix.copy(),
                                                 legal_mask=np.array(board.can_move_dir, dtype=bool))
                    path.append((edge, reward))
                    node = edge.outcomes[key]
                value = self.evaluate_and_expand(node)
                for edge, reward in reversed(path):
                    value = reward + self.gamma * value
                    edge.visit_count += 1
                    edge.value_sum += value
                root.visit_count += 1
        finally:
            self.model.train(was_training)


def softmax_visit_probs(children, temperature):
    """pi(a) proportional to N(a) ** (1/tau), with legal prior fallback."""
    pi = np.zeros(4, dtype=np.float64)
    legal = [a for a, edge in children.items() if edge.prior > 0 or edge.visit_count > 0]
    if not legal:
        return pi.astype(np.float32)
    counts = np.array([children[a].visit_count for a in legal], dtype=np.float64)
    if not counts.any():
        counts = np.array([children[a].prior for a in legal])
    if temperature <= 1e-6:
        pi[legal[int(counts.argmax())]] = 1.0
    else:
        positive = counts > 0
        log_weights = np.full(len(legal), -np.inf)
        log_weights[positive] = np.log(counts[positive]) / temperature
        weights = np.exp(log_weights - log_weights.max())
        pi[legal] = weights / weights.sum()
    return pi.astype(np.float32)


@dataclass
class TrainExample:
    state: np.ndarray
    policy: np.ndarray
    value: float


def self_play_game(model, device, mcts_sims, temperature_moves, seed=0, gamma=1.0):
    mcts = MCTS(model, device, gamma=gamma, seed=seed + 10_000_000)
    env = Gym2048Env()
    obs, info = env.reset(seed=seed)
    behavior_rng = np.random.default_rng(seed + 20_000_000)
    examples, rewards = [], []
    while True:
        root = Node(1.0, legal_mask=np.array(info['can_move_dir']), matrix=obs.copy())
        mcts.run(root, mcts_sims)
        # Retain soft visit targets even after behavior switches to greedy.
        target = softmax_visit_probs(root.children, temperature=1.0)
        behavior = softmax_visit_probs(root.children, 1.0 if len(examples) < temperature_moves else 0.0)
        action = int(behavior_rng.choice(4, p=behavior))
        examples.append(TrainExample(obs.copy(), target, 0.0))
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(float(reward) / REWARD_SCALE)
        if done or truncated:
            break
    for example, value in zip(examples, discounted_returns(rewards, gamma)):
        example.value = float(value)
    return examples, dict(info, steps=len(examples), spawn_return=sum(rewards) * REWARD_SCALE)


def train_policy_value(model, optimizer, batch, device, l2_reg=1e-5):
    pairs = [random.choice(augment_board_and_policy(ex.state, ex.policy)) for ex in batch]
    states = torch.stack([obs_to_tensor(s) for s, _ in pairs]).to(device)
    targets = torch.tensor(np.stack([p for _, p in pairs]), device=device)
    masks = torch.tensor([Board(s).can_move_dir for s, _ in pairs], device=device)
    values_target = torch.tensor([ex.value for ex in batch], device=device)
    model.train()
    logits, values = model(states)
    log_probs = masked_categorical(logits, masks).logits
    policy_loss = -(targets * log_probs).sum(-1).mean()
    value_loss = F.smooth_l1_loss(values, values_target)
    loss = policy_loss + 0.5 * value_loss + l2_reg * sum(p.square().sum() for p in model.parameters())
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(policy_loss.detach()), float(value_loss.detach())


def _self_play_worker(job):
    snapshot, mcts_sims, temperature_moves, seed, gamma = job
    return self_play_game(worker_model(snapshot), torch.device('cpu'), mcts_sims,
                          temperature_moves, seed, gamma)


def collect_self_play(model, device, games, mcts_sims, temperature_moves, seed, gamma, pool):
    if pool.workers > 1:
        snapshot = model_snapshot(model)
        return pool.map(_self_play_worker, [(snapshot, mcts_sims, temperature_moves, seed + i, gamma)
                                           for i in range(games)])
    return [self_play_game(model, device, mcts_sims, temperature_moves, seed + i, gamma)
            for i in range(games)]


def _search_episode(model, seed, mcts_sims, gamma):
    # Seed each game's search independently: worker assignment has no effect.
    mcts = MCTS(model, next(model.parameters()).device, dirichlet_frac=0,
                gamma=gamma, seed=seed)
    def action_fn(state, info, episode_seed):
        root = Node(1.0, matrix=state.copy(), legal_mask=np.array(info['can_move_dir']))
        mcts.run(root, mcts_sims)
        return softmax_visit_probs(root.children, 0).argmax()
    return evaluate(model, 1, seed, action_fn)['results'][0]


def _search_worker(job):
    snapshot, seed, mcts_sims, gamma = job
    return _search_episode(worker_model(snapshot), seed, mcts_sims, gamma)


def evaluate_search(model, episodes=10, seed=1_000_000, mcts_sims=100, gamma=1.0, pool=None):
    if episodes < 1:
        raise ValueError('Evaluation requires at least one episode')
    start = time.perf_counter()
    if pool is not None and pool.workers > 1:
        snapshot = model_snapshot(model)
        results = pool.map(_search_worker, [(snapshot, seed + i, mcts_sims, gamma)
                                            for i in range(episodes)])
    else:
        results = [_search_episode(model, seed + i, mcts_sims, gamma) for i in range(episodes)]
    return summarize_results(results, time.perf_counter() - start)



def augment_board_and_policy(state, policy):
    """All eight D4 symmetries; np.rot90 maps LEFT to DOWN, UP to LEFT."""
    pairs = []
    for flip in (False, True):
        board = np.fliplr(state) if flip else state
        pi = policy[[2, 1, 0, 3]] if flip else policy
        for k in range(4):
            pairs.append((np.rot90(board, k).copy(), np.roll(pi, -k).copy()))
    return pairs


def train(args):
    with TrainingRun(args, 'alphazero') as run:
        model, optimizer, device = run.model, run.optimizer, run.device
        replay = deque(maxlen=args.buffer_size)
        if run.saved:
            replay.extend(TrainExample(**ex) for ex in run.saved.get('replay', []))
        run.ensure_baseline(lambda: evaluate_search(model, args.eval_episodes, args.eval_seed,
                                                   args.eval_mcts_sims, args.gamma, pool=run.pool),
                            {'replay': [vars(ex) for ex in replay]})
        for iteration in range(run.start, args.iterations + 1):
            # 1. Self-play with PUCT search; keep visit targets and remaining returns.
            summaries = []
            started = time.perf_counter()
            games = collect_self_play(model, device, args.self_play_games, args.mcts_sims,
                                      args.temperature_moves,
                                      10_000_000 + args.seed + iteration * 100_000, args.gamma, run.pool)
            for game, (examples, info) in enumerate(games):
                replay.extend(examples)
                summaries.append(info)
                print(json.dumps(dict(iteration=iteration, game=game + 1, steps=len(examples),
                                      max_tile=info['max_value'])), flush=True)
            collect_seconds = time.perf_counter() - started
            started = time.perf_counter()
            # 2. Fit policy and value targets sampled from the replay buffer.
            replay_list = list(replay)
            losses = [train_policy_value(model, optimizer,
                        random.sample(replay_list, min(args.batch_size, len(replay_list))), device)
                      for _ in range(args.train_steps)]
            metrics = dict(collect_seconds=collect_seconds, update_seconds=time.perf_counter() - started,
                           iteration=iteration, replay_size=len(replay), updates=args.train_steps,
                           policy_loss=float(np.mean([p for p, _ in losses])),
                           value_loss=float(np.mean([v for _, v in losses])),
                           train_mean_return=float(np.mean([s['spawn_return'] for s in summaries])),
                           train_mean_steps=float(np.mean([s['steps'] for s in summaries])),
                           train_max_tile=max(s['max_value'] for s in summaries))
            # 3. Validate without exploration noise; checkpoint includes replay for resume.
            if run.should_validate(iteration):
                metrics['validation'] = evaluate_search(model, args.eval_episodes, args.eval_seed,
                                                         args.eval_mcts_sims, args.gamma, pool=run.pool)
            run.record(metrics, {'replay': [vars(ex) for ex in replay]})
        return model


def main(argv=None):
    parser = training_parser(__doc__)
    for flag in ('self-play-games', 'mcts-sims', 'temperature-moves', 'buffer-size',
                 'batch-size', 'train-steps', 'eval-mcts-sims'):
        parser.add_argument('--' + flag, type=int)
    args = resolve_args(parser, 'alphazero', dict(self_play_games=8, mcts_sims=100,
        temperature_moves=30, buffer_size=100_000, batch_size=256, train_steps=100,
        eval_every=10, plot_every=10, eval_mcts_sims=100), argv)
    return train(args)


if __name__ == '__main__':
    main()
