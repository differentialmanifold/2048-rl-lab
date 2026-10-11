"""Coordinate semantics, PPO sampling integrity, and measurable D4 behavior."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from algorithms import ppo
from common.checkpoints import read_checkpoint
from common.models import ActorCritic, masked_categorical, preprocess_observation
from common.parallel import model_snapshot
from common.rollout import Rollout
from common.symmetry import transform_boards, transform_policy, transform_actions
from common.training import setup
from evaluate_symmetry import policy_diagnostics, _continue_boards, paired_difference, require_completed_run
from gym2048_afterstate_env import candidate_afterstates, move_afterstate


def test_all_d4_moves_masks_and_policy_actions_commute():
    rng = np.random.default_rng(19)
    for _ in range(25):
        board = (2 ** rng.integers(0, 8, (4, 4))).astype(np.int32)
        board[board == 1] = 0
        _, mask, _ = candidate_afterstates(board)
        for g in range(8):
            transformed = transform_boards(torch.tensor(board).reshape(16), g).reshape(4, 4).numpy()
            oracle = np.rot90(np.fliplr(board) if g >= 4 else board, g % 4)
            np.testing.assert_array_equal(transformed, oracle)
            _, new_mask, _ = candidate_afterstates(transformed)
            np.testing.assert_array_equal(new_mask, transform_policy(torch.tensor(mask), g))
            for action in range(4):
                mapped_action = int(transform_actions(torch.tensor(action), g))
                assert int(transform_actions(torch.tensor(mapped_action), g, inverse=True)) == action
                original, changed, reward = move_afterstate(board, action)
                actual, new_changed, new_reward = move_afterstate(transformed, mapped_action)
                expected = transform_boards(torch.tensor(original).reshape(16), g).reshape(4, 4)
                np.testing.assert_array_equal(actual, expected)
                assert (changed, reward) == (new_changed, new_reward)
                one_hot = torch.nn.functional.one_hot(torch.tensor(action), 4)
                assert transform_policy(one_hot, g).argmax() == mapped_action


def test_batched_permutations_and_policy_inverse():
    boards = torch.arange(8 * 16).reshape(8, 16)
    policies = torch.arange(8 * 4).reshape(8, 4)
    transforms = torch.arange(8)
    for g in range(8):
        torch.testing.assert_close(transform_boards(boards, transforms)[g], transform_boards(boards[g], g))
    torch.testing.assert_close(transform_policy(transform_policy(policies, transforms), transforms, inverse=True), policies)
    actions = torch.arange(8) % 4
    torch.testing.assert_close(transform_actions(transform_actions(actions, transforms), transforms, inverse=True), actions)


def make_rollout(model):
    boards = [np.array([[2, 2, 4, 0], [8, 16, 0, 0], [32, 0, 0, 0], [64, 128, 0, 0]]),
              np.array([[2, 4, 8, 16], [4, 8, 16, 32], [8, 16, 32, 64], [0, 0, 0, 0]])]
    states = torch.stack([preprocess_observation(b) for b in boards])
    masks = torch.tensor(np.stack([candidate_afterstates(b)[1] for b in boards]))
    with torch.no_grad():
        logits, values = model(states)
        dist = masked_categorical(logits, masks)
        actions = dist.sample()
    return Rollout(states, masks, actions, dist.log_prob(actions), dist.logits,
                   torch.tensor([1., -1.]), values + 1, [])


@pytest.mark.parametrize('architecture', ['cnn2x2', 'vit'])
def test_consistency_losses_detach_teacher_and_backpropagate_student(architecture):
    setup(0)
    model = ActorCritic(architecture=architecture)
    rollout = make_rollout(model)
    logits, values = model(rollout.states)
    logits.retain_grad()
    values.retain_grad()
    dist = masked_categorical(logits, rollout.masks)
    identity = ppo.symmetry_losses(model, rollout.states, rollout.masks, dist, values, torch.zeros(2, dtype=torch.long))
    assert all(abs(float(loss.detach())) < 1e-6 for loss in identity)
    policy_loss, value_loss = ppo.symmetry_losses(model, rollout.states, rollout.masks, dist, values, torch.tensor([1, 4]))
    (policy_loss + value_loss).backward()
    assert logits.grad is None and values.grad is None
    assert model.policy_head.weight.grad.abs().sum() > 0
    assert model.value_head.weight.grad.abs().sum() > 0
    assert torch.isfinite(policy_loss + value_loss)


def test_augmentation_preserves_original_ppo_ratio_targets_and_disabled_rng():
    setup(2)
    original = ActorCritic()
    rollout = make_rollout(original)
    saved = deepcopy(rollout)
    results, rngs = [], []
    for symmetry, coefficient in [('none', .1), ('d4', 0.), ('d4', .1)]:
        setup(12)
        model = deepcopy(original)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.)
        results.append(ppo.update(model, optimizer, rollout, epochs=1, batch_size=2,
            symmetry=symmetry, symmetry_policy_coef=coefficient, symmetry_value_coef=coefficient))
        rngs.append(torch.get_rng_state())
    for key in ('actor', 'critic', 'kl', 'entropy', 'updates'):
        assert results[0][key] == results[1][key] == results[2][key]
    torch.testing.assert_close(rngs[0], rngs[1], atol=0, rtol=0)
    for field in ('states', 'actions', 'masks', 'old_logits', 'old_log_probs', 'returns', 'advantages'):
        torch.testing.assert_close(getattr(rollout, field), getattr(saved, field), atol=0, rtol=0)


class EquivariantPolicy(torch.nn.Module):
    def __init__(self, constant=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.constant = constant

    def forward(self, states):
        boards = states.reshape(-1, 4, 4)
        logits = torch.stack((boards[:, :, 0].sum(-1), boards[:, 0, :].sum(-1),
                              boards[:, :, -1].sum(-1), boards[:, -1, :].sum(-1)), -1)
        return (logits * 0 if self.constant else logits), states.sum(-1)


def test_audit_detects_equivariance_and_fixed_direction_tie_breaking():
    board = [[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 0, 0], [32, 64, 128, 256]]
    exact = policy_diagnostics(EquivariantPolicy(), [board])
    assert exact['mean_pairwise_policy_tv'] < 1e-7
    assert exact['mean_value_spread'] == 0
    assert exact['all_eight_greedy_agreement'] == 1
    tied = policy_diagnostics(EquivariantPolicy(constant=True), [board])
    assert tied['mean_pairwise_policy_tv'] == 0
    assert tied['all_eight_argmax_set_agreement'] == 1
    assert tied['all_eight_greedy_agreement'] == 0


def test_continuations_map_actions_and_score_only_new_rewards():
    setup(0)
    model = ActorCritic()
    # One merge remains; every continuation finishes quickly.
    board = [[2, 2, 8, 16], [16, 8, 4, 2], [2, 4, 8, 16], [16, 8, 4, 2]]
    results = _continue_boards((model_snapshot(model), [(0, board)], 56))
    assert len(results) == 8 and {r['seed'] for r in results} == {56}
    assert all(r['steps'] >= 1 and r['spawn_return'] >= 2 for r in results)
    for row in results:
        assert 2 * row['steps'] <= row['spawn_return'] <= 4 * row['steps']
    comparison = paired_difference([2, 4, 8], [4, 6, 10])
    assert comparison['ci95'] == [2., 2.] and comparison['mean_change'] == 2.


def test_d4_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import plot
    monkeypatch.setattr(plot, 'plot_training', lambda *a, **k: None)
    options = ['--device', 'cpu', '--workers', '1', '--episodes-per-update', '1',
               '--eval-episodes', '1', '--eval-every', '1', '--symmetry', 'd4', '--seed', '7']
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    ppo.main(['--iterations', '2', '--save-dir', str(whole), *options])
    ppo.main(['--iterations', '1', '--save-dir', str(split), *options])
    ppo.main(['--iterations', '2', '--resume', str(split / 'last.pt')])
    expected, actual = read_checkpoint(whole / 'last.pt'), read_checkpoint(split / 'last.pt')
    assert actual['config']['symmetry'] == 'd4'
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], actual['model'][key], atol=0, rtol=0)
    with pytest.raises(SystemExit):
        ppo.main(['--iterations', '3', '--resume', str(whole / 'last.pt'), '--symmetry', 'none'])


@pytest.mark.parametrize('coefficient', ['-1', 'nan', 'inf'])
def test_invalid_symmetry_coefficients_are_rejected(coefficient):
    with pytest.raises(SystemExit):
        ppo.main(['--iterations', '1', '--symmetry-policy-coef', coefficient])


def test_audit_requires_completed_training_but_accepts_early_best_and_published_baseline(tmp_path, monkeypatch):
    import evaluate_symmetry
    best = dict(iteration=15000, model_config={'architecture': 'cnn2x2'}, algorithm='ppo')
    require_completed_run('pretrained/original.pt', dict(best, inference_only=True), 20000)
    with pytest.raises(ValueError, match='missing last.pt'):
        require_completed_run(tmp_path / 'best.pt', best, 20000)
    (tmp_path / 'last.pt').touch()
    final = dict(best, iteration=19999, stop_reason=None)
    monkeypatch.setattr(evaluate_symmetry, 'read_checkpoint', lambda _: final)
    with pytest.raises(ValueError, match='has not completed'):
        require_completed_run(tmp_path / 'best.pt', best, 20000)
    final.update(iteration=20000, stop_reason='iteration_limit')
    require_completed_run(tmp_path / 'best.pt', best, 20000)
    final['stop_reason'] = None
    with pytest.raises(ValueError, match='has not completed'):
        require_completed_run(tmp_path / 'best.pt', best, 20000)
