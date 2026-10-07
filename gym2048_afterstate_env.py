"""An independent 2048 environment with explicit move and chance phases.

Actions remain LEFT, UP, RIGHT, DOWN. A complete step has the same spawn-mass
reward and seeded tile stream as Gym2048Env; candidate afterstates never draw RNG.
"""
import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from board import Board
from gym2048_env import MaskedDiscrete


def move_afterstate(matrix, action):
    """Return (board before spawning, changed, merge reward), without mutation/RNG."""
    if action not in range(4):
        raise ValueError(f'Invalid action {action}; expected 0..3')
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Expected a square board')
    board = Board(matrix, size=len(matrix))
    result = board.matrix
    # Reuse Board's deterministic line merger on oriented writable views.
    board.matrix = (result, result.T, result[:, ::-1], result.T[:, ::-1])[action]
    changed = board.move_left()
    return result.astype(np.int32), bool(changed), float(board.merge_reward)


def candidate_afterstates(matrix):
    """Four candidates in action order, their legal mask, and merge rewards."""
    outcomes = [move_afterstate(matrix, action) for action in range(4)]
    boards, legal, merges = zip(*outcomes)
    return np.stack(boards), np.asarray(legal, dtype=bool), np.asarray(merges)


def spawn_outcomes(afterstate):
    """Enumerate exact chance outcomes; empty cells are uniform, 2/4 are 90%/10%.

    Return (boards, probabilities, spawn rewards). This pure planning helper
    does not sample or affect a real environment's random stream.
    """
    afterstate = np.asarray(afterstate, dtype=np.int32)
    empty = np.argwhere(afterstate == 0)
    if not len(empty):
        raise ValueError('Spawning requires an empty cell')
    boards, probabilities, rewards = [], [], []
    for row, column in empty:
        for value, probability in ((2, 1 - Board.fourProbability), (4, Board.fourProbability)):
            successor = afterstate.copy()
            successor[row, column] = value
            boards.append(successor)
            probabilities.append(probability / len(empty))
            rewards.append(value)
    return np.stack(boards), np.asarray(probabilities), np.asarray(rewards, dtype=np.float32)


class Gym2048AfterstateEnv(gym.Env):
    """step_move(a) -> afterstate; step_spawn() -> next decision state.

    Only step_spawn consumes the tile stream. step(a) composes both phases for
    Gymnasium clients. Illegal moves earn zero and never enter the spawn phase.
    Reset retains the original environment's one initial random tile.
    """
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 60}

    def __init__(self, size=4, render_mode=None, seed=None, matrix=None):
        self.size, self.render_mode = size, render_mode
        self.action_space = MaskedDiscrete(4, mask_getter=self._legal_action_mask)
        self.observation_space = spaces.Box(0, np.iinfo(np.int32).max,
                                            (size, size), dtype=np.int32)
        self._rng = random.Random(seed)
        self.board = Board(matrix, size=size, rng=self._rng)
        self._pending_spawn = False
        self._afterstate = None
        self._spawn_position = None
        self._spawn_value = 0
        if seed is not None:
            self.action_space.seed(seed)

    def _get_obs(self):
        return self.board.matrix.copy().astype(np.int32)

    def _legal_action_mask(self):
        return [False] * 4 if self._pending_spawn else self.board.can_move_dir[:]

    def _get_info(self):
        return dict(can_move_dir=self._legal_action_mask(), max_value=int(self.board.max_value),
                    score=float(self.board.total_score), merge_reward=float(self.board.merge_reward),
                    merge_score=float(self.board.merge_score),
                    phase='afterstate' if self._pending_spawn else 'decision',
                    afterstate=None if self._afterstate is None else self._afterstate.copy(),
                    spawn_position=self._spawn_position, spawn_value=self._spawn_value)

    def _update_status(self):
        self.board.max_value = int(self.board.matrix.max())
        self.board.total_score = float(self.board.matrix.sum()) / 500
        self.board.can_move_dir = self.board.check_swipe_all_direction()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
            self.action_space.seed(seed)
        self.board = Board(size=self.size, rng=self._rng)
        self._pending_spawn = False
        self._afterstate, self._spawn_position, self._spawn_value = None, None, 0
        return self._get_obs(), self._get_info()

    def reset_to_matrix(self, matrix, *, seed=None):
        """Start at a decision state without adding a tile."""
        if seed is not None:
            self._rng.seed(seed)
            self.action_space.seed(seed)
        self.board = Board(matrix, size=self.size, rng=self._rng)
        self._pending_spawn = False
        self._afterstate, self._spawn_position, self._spawn_value = None, None, 0
        return self._get_obs(), self._get_info()

    def legal_actions(self):
        return np.flatnonzero(self._legal_action_mask()).tolist()

    def afterstates(self):
        if self._pending_spawn:
            raise RuntimeError('Complete step_spawn before choosing another move')
        return candidate_afterstates(self.board.matrix)

    def step_move(self, action):
        if self._pending_spawn:
            raise RuntimeError('Complete step_spawn before choosing another move')
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action {action}; expected 0..3')
        afterstate, changed, merge_reward = move_afterstate(self.board.matrix, int(action))
        self.board.matrix = afterstate.astype(int)
        self.board.reward = 0
        self.board.merge_reward = merge_reward
        self.board.merge_score += merge_reward
        self.board.has_changed, self.board.last_action = changed, int(action)
        self._pending_spawn = changed
        self._afterstate = afterstate.copy() if changed else None
        self._spawn_position, self._spawn_value = None, 0
        self._update_status()
        return self._get_obs(), 0., not changed and self.board.has_done(), False, self._get_info()

    def step_spawn(self):
        if not self._pending_spawn:
            raise RuntimeError('step_spawn requires a preceding legal step_move')
        self.board.add_random_tile()
        position = np.argwhere(self.board.matrix != self._afterstate)[0]
        self._spawn_position = tuple(int(x) for x in position)
        self._spawn_value = int(self.board.reward)
        self._pending_spawn = False
        self._update_status()
        if self.render_mode == 'human':
            self.render()
        return (self._get_obs(), float(self.board.reward), self.board.has_done(),
                False, self._get_info())

    def step(self, action):
        result = self.step_move(action)
        return self.step_spawn() if self._pending_spawn else result

    def clone(self):
        clone = Gym2048AfterstateEnv(size=self.size, render_mode=self.render_mode,
                                   matrix=self.board.matrix)
        clone.board = self.board.copy()
        clone._rng = clone.board.rng
        clone._pending_spawn = self._pending_spawn
        clone._afterstate = None if self._afterstate is None else self._afterstate.copy()
        clone._spawn_position, clone._spawn_value = self._spawn_position, self._spawn_value
        return clone

    def render(self):
        return self.board.render_board()

    def close(self):
        pass
