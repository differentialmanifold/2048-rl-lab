import random
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from board import Board


class MaskedDiscrete(spaces.Discrete):
    """Discrete space that samples only from valid actions indicated by a mask.

    The mask is provided by a callable that returns a sequence of booleans of length n.
    """

    def __init__(self, n, mask_getter=None):
        super().__init__(n)
        self._mask_getter = mask_getter

    def sample(self, mask=None):
        if mask is None and self._mask_getter is not None:
            try:
                mask = np.asarray(self._mask_getter(), dtype=bool)
            except Exception:
                mask = None

        if mask is not None:
            valid_indices = np.flatnonzero(mask)
            if valid_indices.size > 0:
                return int(self.np_random.choice(valid_indices))
            # If no valid actions (episode likely terminated), fall back to 0
            return 0

        return super().sample(mask)


class Gym2048Env(gym.Env):
    """
    Gymnasium environment wrapper for the 2048 `Board`.

    Observation: numpy array shape (size, size) with integer tile values.
    Action space: Discrete(4) mapped to 0:LEFT, 1:UP, 2:RIGHT, 3:DOWN.
    Reward: as produced by `Board.step` (value of spawned tile after a valid move).
    Episode termination: no valid moves remain.
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 60,
    }

    def __init__(self, size=4, render_mode=None, seed=None, matrix=None):
        self.size = size
        self.render_mode = render_mode

        # Spaces
        self.action_space = MaskedDiscrete(4, mask_getter=self._legal_action_mask)
        # Allow large integer values; `Board` uses Python ints that can grow.
        self.observation_space = spaces.Box(
            low=0,
            high=np.iinfo(np.int32).max,
            shape=(self.size, self.size),
            dtype=np.int32,
        )

        # Each real environment owns its spawn stream. Planning must not consume it.
        self._rng = random.Random(seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.board = Board(matrix, size=self.size, rng=self._rng)

    def _get_obs(self):
        return self.board.matrix.copy().astype(np.int32)

    def _get_info(self):
        return {
            "can_move_dir": self.board.can_move_dir[:],
            "max_value": int(self.board.max_value),
            "score": float(self.board.total_score),
            "merge_reward": float(self.board.merge_reward),
            "merge_score": float(self.board.merge_score),
        }

    def _legal_action_mask(self):
        # Return mask aligned with action indices 0..3
        return self.board.can_move_dir[:]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset only this environment; leave policy/search RNG streams untouched.
        if seed is not None:
            self._rng.seed(seed)

        self.board = Board(size=self.size, rng=self._rng)
        observation = self._get_obs()
        info = self._get_info()
        # Seed action space for deterministic sampling
        if seed is not None:
            try:
                self.action_space.seed(seed)
            except Exception:
                pass
        return observation, info

    def reset_to_matrix(self, matrix, *, seed=None):
        """Reset environment to a provided board matrix (no random tile added)."""
        if seed is not None:
            self._rng.seed(seed)
            try:
                self.action_space.seed(seed)
            except Exception:
                pass
        self.board = Board(matrix, size=self.size, rng=self._rng)
        return self._get_obs(), self._get_info()

    def legal_actions(self):
        mask = np.asarray(self._legal_action_mask(), dtype=bool)
        return np.flatnonzero(mask).astype(int).tolist()

    def clone(self):
        """Create a lightweight clone of the environment with the same state."""
        new_env = Gym2048Env(size=self.size, matrix=self.board.matrix)
        new_env.board = self.board.copy()
        new_env._rng = new_env.board.rng
        return new_env

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0..3")

        matrix, reward, done, max_value, total_score, can_move_dir = self.board.step(action)
        observation = matrix.astype(np.int32)

        terminated = bool(done)
        truncated = False
        info = {
            "can_move_dir": can_move_dir[:],
            "max_value": int(max_value),
            "score": float(total_score),
            "merge_reward": float(self.board.merge_reward),
            "merge_score": float(self.board.merge_score),
        }

        if self.render_mode == "human":
            self.render()

        return observation, float(reward), terminated, truncated, info

    def render(self):
        return self.board.render_board()

    def close(self):
        pass


__all__ = ["Gym2048Env"]

if __name__ == "__main__":
    # Minimal demo: run a short random episode using the Gym API
    env = Gym2048Env(size=4)
    obs, info = env.reset(seed=0)
    done = False
    step_count = 0
    total_reward = 0.0
    while not done and step_count < 100:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step_count += 1
        print(f"Step {step_count}, action={action}, reward={reward}, terminated={terminated}, can_move_dir={info.get('can_move_dir')}")
        env.render()
    print(f"Episode finished in {step_count} steps, total_reward={total_reward}, max_value={info.get('max_value')}")
