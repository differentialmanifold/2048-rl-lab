"""Gym rollout MCTS: select with UCT, expand, simulate to termination, back up.

No learned model. Search uses terminal board mass / 500 and a fixed rollout count.
"""
import random
import numpy as np
from gym2048_env import Gym2048Env


class TreeNode:
    def __init__(self, matrix: np.ndarray, parent=None, last_action: int = -1):
        self.matrix = np.array(matrix, copy=True)
        self.parent = parent
        self.children: list[TreeNode] = []
        self.visit_count = 0
        self.win_score = 0.0
        self.last_action = last_action

    def _uct_value(self, parent_visit: int, node_win_score: float, node_visit: int) -> float:
        if node_visit == 0:
            return float(2 ** 63)  # large value to force exploration
        return (node_win_score / node_visit) + 1.41 * (np.sqrt(np.log(max(1, parent_visit)) / node_visit))

    def find_best_node_with_uct(self):
        parent_visit = self.visit_count
        return max(self.children, key=lambda child: self._uct_value(parent_visit, child.win_score, child.visit_count))

    def get_random_child_node(self):
        if not self.children:
            return None
        return random.choice(self.children)

    def get_child_with_max_score(self):
        return max(self.children, key=lambda child: child.visit_count)


def _select_promising_node(root_node: TreeNode) -> TreeNode:
    node = root_node
    while len(node.children) > 0:
        node = node.find_best_node_with_uct()
    return node


def _expand_node(promising_node: TreeNode):
    # Construct with matrix once; no separate reset needed (avoids extra RNG)
    env = Gym2048Env(size=4, matrix=promising_node.matrix, seed=random.getrandbits(64))
    info = env._get_info()
    legal_mask = np.asarray(info.get("can_move_dir", [False, False, False, False]), dtype=bool)
    for action in range(4):
        if legal_mask[action]:
            next_env = Gym2048Env(size=4, matrix=promising_node.matrix, seed=random.getrandbits(64))
            next_obs, _, _, _, _ = next_env.step(action)
            child = TreeNode(next_obs, parent=promising_node, last_action=action)
            promising_node.children.append(child)


def _simulate_random_result(node_to_explore: TreeNode) -> float:
    # Construct with matrix once; no separate reset needed (avoids extra RNG)
    env = Gym2048Env(size=4, matrix=node_to_explore.matrix, seed=random.getrandbits(64))
    done = False
    total_score = float(env._get_info().get("score", 0.0))
    while not done:
        mask = np.asarray(env._legal_action_mask(), dtype=bool)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            break
        a = int(random.choice(valid.tolist()))
        _, _, terminated, truncated, info = env.step(a)
        done = bool(terminated or truncated)
        total_score = float(info.get("score", total_score))
    return total_score


def _back_propagation(node_to_explore: TreeNode, playout_result: float):
    temp_node = node_to_explore
    while temp_node is not None:
        temp_node.visit_count += 1
        temp_node.win_score += playout_result
        temp_node = temp_node.parent


def find_next_move(matrix: np.ndarray, budget: int, return_policy=False):
    if budget < 1:
        raise ValueError("Search budget must be positive")
    root = TreeNode(matrix)
    for _ in range(budget):
        promising = _select_promising_node(root)
        _expand_node(promising)
        node_to_explore = promising
        if promising.children:
            node_to_explore = promising.get_random_child_node()
        result = _simulate_random_result(node_to_explore)
        _back_propagation(node_to_explore, result)
    if not root.children:
        # No legal actions
        if return_policy:
            return 0, np.zeros(4, dtype=np.float32)
        return 0
    best = root.get_child_with_max_score()
    if return_policy:
        counts = np.zeros(4, dtype=np.float32)
        for child in root.children:
            counts[child.last_action] = child.visit_count
        return int(best.last_action), counts / counts.sum()
    return int(best.last_action)


class RolloutMCTS:
    """Isolate the demo's planning RNG from callers and real-game spawning."""
    def __init__(self, budget=500, seed=0):
        if budget < 1:
            raise ValueError('Search budget must be positive')
        self.budget = budget
        self.rng = random.Random(seed)

    def search(self, state):
        previous = random.getstate()
        random.setstate(self.rng.getstate())
        try:
            result = find_next_move(state, self.budget, return_policy=True)
            self.rng.setstate(random.getstate())
            return result
        finally:
            random.setstate(previous)


if __name__ == '__main__':
    import argparse
    from play import play, resolve_seed
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=500)
    parser.add_argument('--seed', type=int, help='Optional reproducible seed; random by default')
    args = parser.parse_args()
    args.seed = resolve_seed(args.seed)
    planner = RolloutMCTS(args.budget, args.seed + 10_000_000)
    play(lambda state, info: planner.search(state)[0], args.seed, 'Rollout MCTS')
