"""Student behavior, reverse-KL direction, frozen labels, search and resumption."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from algorithms import opd
from common.checkpoints import save_checkpoint, read_checkpoint, export_model
from common.models import ActorCritic, masked_categorical, preprocess_observation
from common.parallel import GamePool
from common.rollout import collect_actor_critic
from common.training import setup
from gym2048_env import Gym2048Env


@pytest.fixture(autouse=True)
def seed():
    setup(13, 'cpu')


def checkpoint(path, model=None, algorithm='ppo'):
    model = model if model is not None else ActorCritic()
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), 1, algorithm, {}, 0.)
    return model


class ShortEnv(Gym2048Env):
    """Real moves and spawns, truncated to exercise value bootstrap cheaply."""
    def reset(self, **kwargs):
        self.steps = 0
        return super().reset(**kwargs)

    def step(self, action):
        self.steps += 1
        obs, reward, done, _, info = super().step(action)
        return obs, reward, done, self.steps == 3 and not done, info


def test_sampled_expected_gradient_is_reverse_kl_and_labels_are_detached():
    masks = torch.tensor([[True, False, True, True]])
    logits = torch.tensor([[.2, 17., -.8, .9]], requires_grad=True)
    teacher_logits = torch.tensor([[-.3, -20., 1.2, .1]], requires_grad=True)
    teacher_logs = masked_categorical(teacher_logits, masks).logits
    behavior = masked_categorical(logits, masks)
    old_logs = behavior.logits.detach().clone().requires_grad_()
    # Enumerate the action expectation; this catches KL direction, sign, ratio
    # denominator and the otherwise easy-to-miss stop-gradient on the advantage.
    expectation = sum(behavior.probs[0, a].detach() * opd.sampled_distillation_loss(
        behavior, torch.tensor([a]), old_logs[:, a], teacher_logs) for a in (0, 2, 3))
    gradient, = torch.autograd.grad(expectation, logits, retain_graph=True)
    expected, = torch.autograd.grad(opd.reverse_kl(behavior, teacher_logs, masks).mean(), logits)
    torch.testing.assert_close(gradient, expected)
    assert gradient[0, 1] == 0
    assert teacher_logits.grad is None and old_logs.grad is None
    # A second calculation explicitly tests that the sampled loss has no path
    # into teacher logits or saved behavior logits.
    opd.sampled_distillation_loss(masked_categorical(logits, masks), torch.tensor([2]),
                                 old_logs[:, 2], masked_categorical(teacher_logits, masks).logits).backward()
    assert teacher_logits.grad is None and old_logs.grad is None


def test_exact_kl_masks_extreme_logits_and_has_zero_self_loss():
    masks = torch.tensor([[True, False, True, False], [False, True, False, False]])
    logits = torch.tensor([[1000., 10000., -1000., 0.], [0., -1000., 1000., 0.]], requires_grad=True)
    dist = masked_categorical(logits, masks)
    teacher = masked_categorical(-logits.detach(), masks)
    loss = opd.reverse_kl(dist, teacher.logits, masks)
    assert loss[0] == 2000 and loss[1] == 0
    loss.mean().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[~masks], torch.zeros_like(logits.grad[~masks]))
    torch.testing.assert_close(opd.reverse_kl(dist, dist.logits.detach(), masks), torch.zeros(2))


def test_frozen_ppo_labels_use_legal_mask_and_exact_update_reduces_kl():
    student, teacher = ActorCritic(), ActorCritic(architecture='vit')
    with torch.no_grad():
        teacher.policy_head.bias.copy_(torch.tensor([1., 0., -1., 2.]))
    before = deepcopy(teacher.state_dict())
    env = ShortEnv()
    rollout = collect_actor_critic(env, student, 2, .999, 8)
    labels = opd.ppo_teacher_log_probs(teacher, rollout.states, rollout.masks, batch_size=2)
    assert not labels.requires_grad
    assert torch.equal(labels.exp()[~rollout.masks], torch.zeros_like(labels[~rollout.masks]))
    dist = masked_categorical(student(rollout.states)[0], rollout.masks)
    initial = opd.reverse_kl(dist, labels, rollout.masks).mean().item()
    metrics = opd.update(student, torch.optim.Adam(student.parameters(), lr=.001), rollout,
                         labels, epochs=10, batch_size=6, loss_kind='exact', value_coef=0., target_kl=0.)
    after = opd.reverse_kl(masked_categorical(student(rollout.states)[0], rollout.masks),
                           labels, rollout.masks).mean().item()
    assert after < initial and metrics['updates'] == 10
    assert all(p.grad is None for p in teacher.parameters())
    for key, weight in teacher.state_dict().items():
        torch.testing.assert_close(before[key], weight, atol=0, rtol=0)
    env.close()


def test_policy_update_ignores_environment_advantages_and_uses_saved_behavior():
    student, teacher = ActorCritic(), ActorCritic()
    env = ShortEnv()
    rollout = collect_actor_critic(env, student, 1, .999, 19)
    labels = opd.ppo_teacher_log_probs(teacher, rollout.states, rollout.masks)
    alternative = deepcopy(rollout)
    alternative.advantages.fill_(1e6)
    alternative.returns.fill_(-1e6)
    copy = deepcopy(student)
    rng = torch.get_rng_state()
    opd.update(student, torch.optim.Adam(student.parameters()), rollout, labels,
               epochs=1, value_coef=0.)
    torch.set_rng_state(rng)
    opd.update(copy, torch.optim.Adam(copy.parameters()), alternative, labels,
               epochs=1, value_coef=0.)
    for key, value in student.state_dict().items():
        torch.testing.assert_close(value, copy.state_dict()[key], atol=0, rtol=0)
    env.close()


def test_search_uses_current_model_and_smooths_only_legal_actions(monkeypatch):
    model = ActorCritic()
    state = np.array([[2, 0, 0, 0], *[[0] * 4] * 3])
    states = preprocess_observation(state)[None]
    masks = torch.tensor([[False, False, True, True]])
    calls = []

    def search(self, root, sims):
        assert self.model is model and self.dirichlet_frac == 0
        np.testing.assert_array_equal(root.matrix, state)
        calls.append((sims, self.gamma))
        root.children = {2: opd.Node(.5, visit_count=1), 3: opd.Node(.5)}

    monkeypatch.setattr(opd.MCTS, 'run', search)
    before = deepcopy(model.state_dict())
    labels = opd.search_teacher_log_probs(model, states, masks, 9, mcts_sims=1,
                                          gamma=.99, search_smoothing=.1)
    torch.testing.assert_close(labels.exp(), torch.tensor([[0., 0., .95, .05]]))
    assert calls == [(1, .99)] and not labels.requires_grad
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert model.training


def test_real_search_matches_workers_preserves_rng_and_changes_with_student():
    model = ActorCritic()
    board = np.array([[0, 0, 0, 0], [0, 2, 0, 0], *[[0] * 4] * 2])
    states = torch.stack([preprocess_observation(board)] * 4)
    masks = torch.ones(4, 4, dtype=torch.bool)
    with torch.no_grad():
        model.policy_head.weight.zero_()
        model.policy_head.bias.copy_(torch.tensor([20., -20., -20., -20.]))
    rng = torch.get_rng_state().clone()
    first = opd.search_teacher_log_probs(model, states, masks, 31, mcts_sims=2)
    with GamePool(2) as pool:
        parallel = opd.search_teacher_log_probs(model, states, masks, 31, mcts_sims=2, pool=pool)
    torch.testing.assert_close(first, parallel, atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert torch.isfinite(first).all() and (first.exp() > 0).all()
    with torch.no_grad():
        model.policy_head.bias.copy_(torch.tensor([-20., -20., 20., -20.]))
    second = opd.search_teacher_log_probs(model, states, masks, 31, mcts_sims=2)
    assert not torch.equal(first, second)


@pytest.mark.parametrize('teacher_mode', ['ppo', 'alphazero'])
@pytest.mark.parametrize('loss', ['sampled', 'exact'])
def test_resume_matches_uninterrupted_and_embeds_frozen_teacher(tmp_path, monkeypatch, teacher_mode, loss):
    import plot
    monkeypatch.setattr(plot, 'plot_training', lambda *a, **k: None)
    monkeypatch.setattr(opd, 'Gym2048Env', ShortEnv)
    monkeypatch.setattr(opd, 'evaluate', lambda *a, **k: {'mean_return': 1.})
    options = ['--device', 'cpu', '--workers', '1', '--teacher', teacher_mode, '--loss', loss,
               '--episodes-per-update', '2', '--eval-episodes', '1', '--eval-every', '1',
               '--epochs', '2', '--batch-size', '4', '--mcts-sims', '2']
    teacher_path = tmp_path / 'teacher.pt'
    if teacher_mode == 'ppo':
        checkpoint(teacher_path)
        options += ['--teacher-checkpoint', str(teacher_path)]
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    opd.main(['--iterations', '2', '--save-dir', str(whole), *options])
    opd.main(['--iterations', '1', '--save-dir', str(split), *options])
    if teacher_mode == 'ppo':
        teacher_path.unlink()  # Resume must need no external teacher file.
    opd.main(['--iterations', '2', '--resume', str(split / 'last.pt'), '--workers', '1'])
    expected, actual = read_checkpoint(whole / 'last.pt'), read_checkpoint(split / 'last.pt')
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], actual['model'][key], atol=0, rtol=0)
    assert actual['config']['teacher'] == teacher_mode and actual['config']['loss'] == loss
    assert ('distillation_teacher' in actual) == (teacher_mode == 'ppo')
    if teacher_mode == 'ppo':
        for key, value in expected['distillation_teacher']['model'].items():
            torch.testing.assert_close(value, actual['distillation_teacher']['model'][key], atol=0, rtol=0)
        with pytest.raises(SystemExit):
            opd.main(['--iterations', '3', '--resume', str(split / 'last.pt'),
                      '--teacher-checkpoint', str(tmp_path / 'different.pt')])
    with pytest.raises(SystemExit):
        opd.main(['--iterations', '3', '--resume', str(split / 'last.pt'),
                  '--init-checkpoint', str(split / 'last.pt')])
    assert 'replay' not in actual
    exported = tmp_path / 'export.pt'
    export_model(split / 'last.pt', exported)
    assert 'distillation_teacher' not in read_checkpoint(exported)
    from play import make_agent
    agent, label, metadata = make_agent('opd', exported)
    board = np.array([[2, 0, 0, 0], *[[0] * 4] * 3])
    assert agent(board, {'can_move_dir': [False, False, True, True]}) in (2, 3)
    assert 'OPD' in label and metadata['search_budget'] == 0


def test_student_samples_actions_even_when_teacher_disagrees(tmp_path, monkeypatch):
    import plot
    monkeypatch.setattr(plot, 'plot_training', lambda *a, **k: None)
    monkeypatch.setattr(opd, 'evaluate', lambda *a, **k: {'mean_return': 1.})
    actions = []

    class OneMove:
        def reset(self, seed):
            return np.full((4, 4), 2), {'can_move_dir': [True] * 4}

        def step(self, action):
            actions.append(action)
            return np.full((4, 4), 2), 2., True, False, {'can_move_dir': [False] * 4, 'max_value': 2}

        def close(self):
            pass

    monkeypatch.setattr(opd, 'Gym2048Env', OneMove)
    student, teacher = ActorCritic(), ActorCritic()
    for model, bias in ((student, [40., -40., -40., -40.]), (teacher, [-40., -40., 40., -40.])):
        with torch.no_grad():
            model.policy_head.weight.zero_()
            model.policy_head.bias.copy_(torch.tensor(bias))
    checkpoint(tmp_path / 'student.pt', student)
    checkpoint(tmp_path / 'teacher.pt', teacher)
    opd.main(['--iterations', '1', '--teacher', 'ppo', '--teacher-checkpoint', str(tmp_path / 'teacher.pt'),
              '--init-checkpoint', str(tmp_path / 'student.pt'), '--device', 'cpu', '--workers', '1',
              '--episodes-per-update', '1', '--save-dir', str(tmp_path / 'out')])
    assert actions == [0]


@pytest.mark.parametrize('options', [[], ['--teacher', 'alphazero', '--value-coef', '0'],
    ['--teacher', 'alphazero', '--teacher-checkpoint', 'ignored.pt'],
    ['--teacher', 'alphazero', '--search-smoothing', '0'],
    ['--teacher', 'alphazero', '--c-puct', 'nan'],
    ['--teacher', 'alphazero', '--target-kl', '-1'],
    ['--teacher', 'alphazero', '--search-temperature', 'inf'],
    ['--teacher', 'alphazero', '--architecture', 'vit']])
def test_invalid_cli_fails_before_training(options):
    with pytest.raises(SystemExit):
        opd.main(['--iterations', '1', *options])


def test_rejects_non_ppo_teacher_and_non_cnn_student(tmp_path):
    checkpoint(tmp_path / 'wrong.pt', algorithm='alphazero')
    with pytest.raises(ValueError, match='PPO checkpoint'):
        opd.main(['--iterations', '1', '--teacher-checkpoint', str(tmp_path / 'wrong.pt')])
    checkpoint(tmp_path / 'vit.pt', model=ActorCritic(architecture='vit'))
    with pytest.raises(ValueError, match='cnn2x2'):
        opd.main(['--iterations', '1', '--teacher', 'alphazero',
                  '--init-checkpoint', str(tmp_path / 'vit.pt')])
