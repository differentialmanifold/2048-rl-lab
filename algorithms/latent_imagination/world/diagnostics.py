"""Read-only model diagnostics. Exact board rules are used ONLY as an oracle here."""

import numpy as np
import torch


def raw_board(ranks):
    ranks = np.asarray(ranks, dtype=np.int64).reshape(4, 4)
    return np.where(ranks == 0, 0, np.left_shift(1, ranks))


def outcome_distribution(ranks, action):
    from board import Board
    from common.models import preprocess_observation
    board = Board(raw_board(ranks))
    lines = (board.matrix, board.matrix.T, board.matrix[:, ::-1], board.matrix.T[:, ::-1])[action]
    changed = board._move_lines(lines)
    if not changed:
        return {tuple(np.asarray(ranks).reshape(-1).tolist()): 1.}, {}
    result, fours = {}, {}
    empty = np.argwhere(board.matrix == 0)
    for row, col in empty:
        for value, probability in ((2, 1 - Board.fourProbability), (4, Board.fourProbability)):
            matrix = board.matrix.copy()
            matrix[row, col] = value
            key = tuple(preprocess_observation(matrix).long().tolist())
            result[key] = probability / len(empty)
            if value == 4:
                fours[key] = result[key]
    return result, fours


@torch.no_grad()
def one_step_diagnostics(model, batch):
    world = model.world
    z = world.encode(batch['states'])
    reconstruction = world.tokenizer.decode(z).argmax(-1)
    u = world.afterstate(z, batch['actions'])
    probability = world.event_probabilities(z, u, batch['actions']).cpu().double().numpy()
    probability /= probability.sum(-1, keepdims=True)
    decoded = world.tokenizer.decode(world.all_events(u).flatten(0, 1)).argmax(-1)
    decoded = decoded.reshape(len(z), world.events, 16).cpu().numpy()
    states, actions = batch['states'].cpu().numpy(), batch['actions'].cpu().numpy()
    invalid, tv, four_mass, reference_four = [], [], [], []
    for state, action, predictions, probs in zip(states, actions, decoded, probability):
        true, fours = outcome_distribution(state, int(action))
        generated = {}
        for ranks, p in zip(predictions, probs):
            key = tuple(ranks.tolist())
            generated[key] = generated.get(key, 0.) + float(p)
        bad = sum(p for key, p in generated.items() if key not in true)
        invalid.append(bad)
        tv.append(.5 * (bad + sum(abs(p - generated.get(key, 0.)) for key, p in true.items())))
        four_mass.append(sum(generated.get(key, 0.) for key in fours))
        reference_four.append(sum(fours.values()))
    return dict(reconstruction_board_accuracy=float((reconstruction == batch['states']).all(-1).float().mean()),
        decoded_invalid_mass=float(np.mean(invalid)), decoded_branch_tv=float(np.mean(tv)),
        decoded_spawn4_mass=float(np.mean(four_mass)), reference_spawn4_mass=float(np.mean(reference_four)))
