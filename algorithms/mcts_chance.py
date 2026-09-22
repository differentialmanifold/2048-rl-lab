"""Rollout MCTS with resampled chance outcomes for stochastic 2048.

UCT chooses actions, while Gym samples tile spawns on EVERY traversal. Each
action can lead to several observed boards, whose rollout returns are averaged
by their actual sampling frequency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

import numpy as np

from gym2048_env import Gym2048Env


@dataclass
class ActionEdge:
    visits: int = 0
    value_sum: float = 0.
    outcomes: dict[bytes, StateNode] = field(default_factory=dict)

    @property
    def mean_value(self):
        return self.value_sum / self.visits if self.visits else 0.


@dataclass
class StateNode:
    matrix: np.ndarray
    actions: dict[int, ActionEdge]
    visits: int = 0

    @classmethod
    def from_observation(cls, observation, info):
        return cls(np.array(observation, copy=True),
                   {action: ActionEdge() for action, legal in enumerate(info['can_move_dir']) if legal})


class ChanceMCTS:
    """UCT action selection, sampled outcomes and complete random legal rollouts."""
    def __init__(self, budget=500, seed=0, exploration=1.41):
        if budget < 1:
            raise ValueError('Search budget must be positive')
        if exploration < 0 or not math.isfinite(exploration):
            raise ValueError('Exploration must be finite and nonnegative')
        self.budget, self.exploration = budget, exploration
        self.rng = random.Random(seed)
        self.last_root = None
        self.last_stats = {}

    def _select_action(self, node):
        # Visit every legal action before exploiting its estimated mean return.
        unvisited = [action for action, edge in node.actions.items() if edge.visits == 0]
        if unvisited:
            return self.rng.choice(unvisited)
        log_visits = math.log(max(1, node.visits))
        return max(node.actions, key=lambda action:
                   node.actions[action].mean_value
                   + self.exploration * math.sqrt(log_visits / node.actions[action].visits))

    def _rollout(self, env, info):
        """Uniform legal actions until true termination, using the simulation env."""
        while True:
            actions = [action for action, legal in enumerate(info['can_move_dir']) if legal]
            if not actions:
                return float(info['score'])  # Terminal mass / 500; root mass is constant, so this maximizes future spawn mass.
            _, _, done, truncated, info = env.step(self.rng.choice(actions))
            self.last_stats['rollout_transitions'] += 1
            if done or truncated:
                return float(info['score'])

    def search(self, state):
        # The real game is never stepped or reseeded. This env belongs to search.
        env = Gym2048Env(size=4, matrix=state, seed=self.rng.getrandbits(64))
        info = env._get_info()
        root = StateNode.from_observation(state, info)
        self.last_root = root
        self.last_stats = dict(simulations=0, tree_transitions=0, rollout_transitions=0)
        try:
            if not root.actions:
                return 0, np.zeros(4, dtype=np.float32)
            for _ in range(self.budget):
                # Reset the board only. The simulation RNG keeps advancing,
                # so visits do not replay one frozen tile-spawn sequence.
                _, info = env.reset_to_matrix(root.matrix)
                node, nodes, edges = root, [root], []
                while node.actions:
                    action = self._select_action(node)
                    edge = node.actions[action]
                    observation, _, done, truncated, info = env.step(action)
                    self.last_stats['tree_transitions'] += 1
                    edges.append(edge)
                    # Sampling occurs before this lookup, including on revisits.
                    # Never maximize UCT over outcomes: nature chooses those.
                    key = observation.tobytes()
                    new_outcome = key not in edge.outcomes
                    if new_outcome:
                        edge.outcomes[key] = StateNode.from_observation(observation, info)
                    node = edge.outcomes[key]
                    nodes.append(node)
                    if new_outcome or done or truncated:
                        break
                value = self._rollout(env, info)
                # One complete terminal return per visited action, not an
                # unweighted average over the distinct outcome boards.
                for edge in edges:
                    edge.visits += 1
                    edge.value_sum += value
                for visited in nodes:
                    visited.visits += 1
                self.last_stats['simulations'] += 1
            counts = np.zeros(4, dtype=np.float32)
            for action, edge in root.actions.items():
                counts[action] = edge.visits
            self.last_stats['root_outcomes'] = {action: len(edge.outcomes)
                                               for action, edge in root.actions.items()}
            return int(counts.argmax()), counts / counts.sum()
        finally:
            env.close()


if __name__ == '__main__':
    import argparse
    from play import play, resolve_seed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=500)
    parser.add_argument('--seed', type=int, help='Optional reproducible seed; random by default')
    args = parser.parse_args()
    if args.budget < 1:
        parser.error('Search budget must be positive')
    seed = resolve_seed(args.seed)
    planner = ChanceMCTS(args.budget, seed + 10_000_000)
    play(lambda state, info: planner.search(state)[0], seed, 'Chance MCTS')
