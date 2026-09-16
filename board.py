import random
import numpy as np

LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3


class Board:
    action_space = 4
    fourProbability = 0.1

    def __init__(self, matrix=None, size=4, rng=None):
        self.size = size
        self.rng = rng if rng is not None else random
        self.merge_reward = 0
        self.merge_score = 0
        if matrix is None:
            self.matrix = np.zeros((self.size, self.size), dtype=int)
            self.add_random_tile()
        else:
            # Initialize from provided matrix-like (numpy array or list of lists)
            self.matrix = np.array(matrix, dtype=int).reshape((self.size, self.size))
        self.has_changed = True
        # Initialize statistics based on current matrix
        self.max_value = int(np.max(self.matrix))
        self.total_score = float(np.sum(self.matrix)) / 500
        self.lost = False
        self.can_move_dir = self.check_swipe_all_direction()
        self.last_action = -1
        self.reward = 0
        self.num_status = 0

    def copy(self):
        rng = self.rng
        if isinstance(rng, random.Random):
            rng = random.Random()
            rng.setstate(self.rng.getstate())
        board_copy = Board(np.array(self.matrix, copy=True), size=self.size, rng=rng)
        board_copy.has_changed = self.has_changed
        board_copy.max_value = self.max_value
        board_copy.total_score = self.total_score
        board_copy.lost = self.lost
        board_copy.can_move_dir = self.can_move_dir[:]
        board_copy.merge_reward = self.merge_reward
        board_copy.merge_score = self.merge_score
        return board_copy

    def _move_lines(self, lines):
        """Slide/merge writable row or column views toward their first element."""
        has_changed = False
        for line in lines:
            original = line.tolist()
            tiles = [value for value in original if value != 0]
            merged = []
            index = 0
            while index < len(tiles):
                value = tiles[index]
                if index + 1 < len(tiles) and tiles[index + 1] == value:
                    # Consume both tiles: a newly merged tile cannot merge again.
                    value *= 2
                    self.merge_reward += value
                    index += 2
                else:
                    index += 1
                merged.append(value)
            merged.extend([0] * (self.size - len(merged)))
            if merged != original:
                has_changed = True
                line[:] = merged
        return has_changed

    def move_left(self):
        return self._move_lines(self.matrix)

    def add_random_tile(self):
        empty_positions = np.argwhere(self.matrix == 0)
        if empty_positions.size == 0:
            return
        index = self.rng.choice(range(len(empty_positions)))
        row, column = empty_positions[index]
        new_value = 4 if self.rng.random() < Board.fourProbability else 2
        self.matrix[row][column] = new_value
        self.reward = new_value

    def move(self, direction):
        self.reward = 0
        self.merge_reward = 0

        # Rows/columns are writable views, ordered from the destination edge.
        # Transpose selects columns; reversing each line selects right/bottom.
        if direction == LEFT:
            lines = self.matrix
        elif direction == UP:
            lines = self.matrix.T
        elif direction == RIGHT:
            lines = self.matrix[:, ::-1]
        elif direction == DOWN:
            lines = self.matrix.T[:, ::-1]
        else:
            raise ValueError(f'Invalid direction {direction}; expected 0..3')
        has_changed = self._move_lines(lines)
        if has_changed:
            self.add_random_tile()
        self.last_action = direction
        self.has_changed = has_changed
        self.can_move_dir = self.check_swipe_all_direction()
        self.merge_score += self.merge_reward

    def can_swipe_left(self):
        return any(right != 0 and (left == 0 or left == right)
                   for row in self.matrix.tolist() for left, right in zip(row, row[1:]))

    def check_swipe_all_direction(self):
        """Check adjacent horizontal/vertical pairs without changing the board.

        An equal nonzero pair allows movement both ways. A tile beside an empty
        cell allows movement toward that cell. Any longer slide contains such
        a pair, so simulating merges is unnecessary for the legality mask.
        """
        rows = self.matrix.tolist()
        left = up = right = down = False
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                if c:
                    previous = row[c - 1]
                    left |= value != 0 and (previous == 0 or previous == value)
                    right |= previous != 0 and (value == 0 or previous == value)
                if r:
                    previous = rows[r - 1][c]
                    up |= value != 0 and (previous == 0 or previous == value)
                    down |= previous != 0 and (value == 0 or previous == value)
            if left and up and right and down:
                break
        return [left, up, right, down]

    def has_lost(self):
        return not any(self.can_move_dir)

    def has_done(self):
        return self.has_lost()

    def env_init(self):
        self.__init__(size=self.size, rng=self.rng)
        _matrix = self.matrix
        return _matrix, self.can_move_dir

    def reset(self):
        return self.env_init()

    def step(self, _action):
        self.move(_action)
        _matrix = self.matrix
        _done = False

        if self.has_done():
            _done = True

        self.max_value = np.max(_matrix)

        self.total_score = np.sum(_matrix) / 500

        return _matrix, self.reward, _done, self.max_value, self.total_score, self.can_move_dir

    def render_board(self):
        _m = self.matrix
        # Use fixed width of 5 for alignment
        cell_width = 5
        lines = ['---']
        for i in range(self.size):
            row_str = []
            for j in range(self.size):
                val = int(_m[i][j])
                row_str.append(str(val).rjust(cell_width))
            lines.append(' '.join(row_str))
        lines.append('---')
        return '\n'.join(lines)

if __name__ == "__main__":
    board = Board()
    print(board.matrix)
    board.move(1)
    print(board.matrix)
    board.move(2)
    print(board.matrix)
    new_board = Board(board.matrix)
    print(new_board.matrix)
