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

    def rotate_left(self):
        # Rotate the board 90 degrees counter-clockwise (left)
        self.matrix = np.rot90(self.matrix)

    def move_left(self):
        has_changed = False
        for row in range(self.size):
            original_row = self.matrix[row].tolist()
            non_zero_values = [value for value in original_row if value != 0]
            merged_row = []
            index = 0
            while index < len(non_zero_values):
                current_value = non_zero_values[index]
                if index + 1 < len(non_zero_values) and non_zero_values[index + 1] == current_value:
                    # Merge equal adjacent tiles
                    merged_row.append(current_value + non_zero_values[index + 1])
                    self.merge_reward += 2 * current_value
                    index += 2
                else:
                    merged_row.append(current_value)
                    index += 1
            # Pad with zeros to the right
            while len(merged_row) < self.size:
                merged_row.append(0)
            if merged_row != original_row:
                has_changed = True
            self.matrix[row] = np.array(merged_row, dtype=int)
        return has_changed

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

        # 0 -> left, 1 -> up, 2 -> right, 3 -> down
        for _ in range(direction):
            self.rotate_left()
        has_changed = self.move_left()
        for _ in range(direction, 4):
            self.rotate_left()
        if has_changed:
            self.add_random_tile()
        self.last_action = direction
        self.has_changed = has_changed
        self.can_move_dir = self.check_swipe_all_direction()
        self.merge_score += self.merge_reward

    def can_swipe_left(self):
        """
        compare adjacent cells
        two condition can move:
        1. first cell is empty and second cell is not empty
        2. two cells not empty and value is equal
        """
        for row in range(self.size):
            for column in range(self.size - 1):
                first_value = int(self.matrix[row][column])
                second_value = int(self.matrix[row][column + 1])
                if first_value == 0 and second_value != 0:
                    return True
                if first_value != 0 and second_value != 0 and first_value == second_value:
                    return True
        return False

    def check_swipe_all_direction(self):
        can_move_dir = [False, False, False, False]
        # left, up, right, down
        for i in range(4):
            can_move_dir[i] = self.can_swipe_left()
            self.rotate_left()
        return can_move_dir

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
