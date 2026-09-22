import json
from pathlib import Path
import numpy as np
import pytest
import torch
from common.models import ActorCritic, masked_categorical
from common.checkpoints import save_checkpoint, load_checkpoint, model_from_checkpoint, read_checkpoint
from common.training import setup
from plot import read_metrics, moving_average
from algorithms import a2c, ppo, alphazero


def test_cnn2x2_batch_mask_consistency_gradient_and_checkpoint(tmp_path):
    setup(6)
    model = ActorCritic(architecture='cnn2x2')
    x = torch.arange(16).float()
    batch_logits, batch_values = model(torch.stack([x, x + 1]))
    logits, value = model(x)
    torch.testing.assert_close(logits, batch_logits[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(value, batch_values[0], atol=1e-6, rtol=1e-6)
    mask = [True, False, True, False]
    old = masked_categorical(logits.detach(), mask)
    dist = masked_categorical(batch_logits[0], mask)
    torch.testing.assert_close(dist.logits[[0, 2]], old.logits[[0, 2]], atol=1e-6, rtol=1e-6)
    (-dist.log_prob(torch.tensor(0)) + value.square()).backward()
    assert model.policy_head.weight.grad.abs().sum() > 0
    assert model.trunk[0].layers[0].weight.grad.abs().sum() > 0
    model.eval()
    torch.testing.assert_close(model(x)[0], logits)
    path = tmp_path / 'cnn.pt'
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), 1, 'ppo', {}, 1)
    restored = model_from_checkpoint(read_checkpoint(path))
    assert restored.architecture == 'cnn2x2'
    torch.testing.assert_close(restored(x)[0], model(x)[0])
    with pytest.raises(ValueError, match='different architecture'):
        load_checkpoint(path, ActorCritic(architecture='vit'))




def test_plot_reading_preserves_real_iterations_and_missing_values(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    path.write_text('\n'.join([json.dumps({'iteration': 1}), json.dumps({'iteration': 100}), '{unfinished']))
    with pytest.warns(UserWarning, match='incomplete'):
        rows = read_metrics(path)
    assert [r['iteration'] for r in rows] == [1, 100]
    np.testing.assert_allclose(moving_average([1, np.nan, 3, 5], 2), [1, 1, 3, 4])


def test_periodic_plot_and_final_validation(tmp_path, monkeypatch):
    import plot
    plotted = []
    monkeypatch.setattr(plot, 'plot_training', lambda logs, **kwargs:
                        plotted.append(read_metrics(logs)[-1]['iteration']))
    directory = tmp_path / 'a2c'
    a2c.main(['--device', 'cpu', '--iterations', '3', '--episodes-per-update', '1', '--eval-every', '2',
              '--eval-episodes', '1', '--plot-every', '2', '--save-dir', str(directory)])
    rows = read_metrics(directory / 'metrics.jsonl')
    assert plotted == [2, 3]  # Plot before the run ends, and again at its final iteration.
    assert [r['iteration'] for r in rows if 'validation' in r] == [2, 3]
    assert all(r['updates'] == 1 for r in rows)
    assert rows[-1]['stop_reason'] == 'iteration_limit'


@pytest.mark.parametrize('trainer', [a2c, ppo])
def test_fixed_iteration_resume_matches_uninterrupted(tmp_path, monkeypatch, trainer):
    import plot
    monkeypatch.setattr(plot, 'plot_training', lambda *a, **k: None)
    options = ['--device', 'cpu', '--episodes-per-update', '1', '--eval-episodes', '1', '--eval-every', '1', '--seed', '7']
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    trainer.main(['--iterations', '3', '--save-dir', str(whole), *options])
    trainer.main(['--iterations', '2', '--save-dir', str(split), *options])
    # No repeated hyperparameters: resume inherits the checkpoint configuration.
    trainer.main(['--iterations', '3', '--resume', str(split / 'last.pt')])
    expected, resumed = read_checkpoint(whole / 'last.pt'), read_checkpoint(split / 'last.pt')
    assert resumed['iteration'] == 3 and resumed['config']['seed'] == 7
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], resumed['model'][key], atol=0, rtol=0)


def test_resume_branch_baseline_and_rewind_guard(tmp_path, monkeypatch):
    import plot
    monkeypatch.setattr(plot, 'plot_training', lambda *a, **k: None)
    monkeypatch.setattr(ppo, 'evaluate', lambda *a, **k: {'mean_return': 100.})
    model = ActorCritic()
    source = tmp_path / 'source.pt'
    save_checkpoint(source, model, torch.optim.Adam(model.parameters()), 1, 'ppo',
                    {'episodes_per_update': 1, 'eval_episodes': 1, 'device': 'cpu', 'td_steps': 10, 'td_lambda': .5}, 9999)
    branch = tmp_path / 'branch'
    args = ['--iterations', '2', '--resume', str(source), '--save-dir', str(branch)]
    ppo.main(args)
    best = read_checkpoint(branch / 'best.pt')
    assert best['best_metric'] == 100 and best['iteration'] == 1
    with pytest.raises(ValueError, match='older than existing logs'):
        ppo.main(args)


def test_exported_models_keep_predictions_but_cannot_resume(tmp_path):
    from common.checkpoints import export_model
    from play import make_agent
    source, exported = tmp_path / 'last.pt', tmp_path / 'demo.pt'
    model = ActorCritic(architecture='cnn2x2')
    save_checkpoint(source, model, torch.optim.Adam(model.parameters()), 1, 'ppo', {}, 0)
    export_model(source, exported)
    data = torch.load(exported, weights_only=True)
    assert 'optimizer' not in data and 'rng_py' not in data
    restored = model_from_checkpoint(data)
    x = torch.arange(16).float()
    torch.testing.assert_close(model(x)[0], restored(x)[0], atol=0, rtol=0)
    agent, label, _ = make_agent('ppo', exported)
    board = np.zeros((4, 4), dtype=int)
    board[0, 0] = 2
    assert agent(board, {'can_move_dir': [False, False, True, True]}) in [2, 3]
    assert 'CNN2X2' in label
    with pytest.raises(ValueError, match='full training checkpoint'):
        ppo.main(['--iterations', '2', '--resume', str(exported)])


@pytest.mark.parametrize('trainer', [a2c, ppo, alphazero])
def test_removed_plateau_flag_is_rejected(trainer):
    with pytest.raises(SystemExit):
        trainer.main(['--iterations', '1', '--until-plateau'])




def test_alphazero_resume_keeps_replay_and_matches_uninterrupted(tmp_path, monkeypatch):
    import plot
    monkeypatch.setattr(plot, 'plot_training', lambda *a, **k: None)
    options = ['--device', 'cpu', '--self-play-games', '1', '--mcts-sims', '2', '--train-steps', '1',
               '--batch-size', '16', '--eval-every', '1', '--eval-episodes', '1', '--eval-mcts-sims', '2']
    whole, split = tmp_path / 'whole', tmp_path / 'split'
    alphazero.main(['--iterations', '2', '--save-dir', str(whole), *options])
    alphazero.main(['--iterations', '1', '--save-dir', str(split), *options])
    first = read_checkpoint(split / 'last.pt')
    alphazero.main(['--iterations', '2', '--resume', str(split / 'last.pt')])
    expected, resumed = read_checkpoint(whole / 'last.pt'), read_checkpoint(split / 'last.pt')
    assert resumed['reward_objective'] == 'spawn_mass'
    assert len(resumed['replay']) > len(first['replay'])
    assert len(resumed['replay']) == len(expected['replay'])
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], resumed['model'][key], atol=0, rtol=0)


def test_interactive_seed_defaults_to_os_randomness(monkeypatch):
    import play
    draws = iter([12345, 98765])
    monkeypatch.setattr(play.secrets, 'randbits', lambda bits: next(draws))
    assert play.resolve_seed() == 12345
    assert play.resolve_seed(0) == 0  # Explicit reproducibility does not consume a draw.
    assert play.resolve_seed() == 98765


def test_plot_contains_all_four_tile_thresholds(tmp_path, monkeypatch):
    from plot import plot_training
    from matplotlib.axes import Axes
    values = {'≥ 512': 100., '≥ 1024': 60., '≥ 2048': 20., '≥ 4096': 10.}
    recorded = {}
    original = Axes.plot
    def capture(self, *args, **kwargs):
        label = kwargs.get('label')
        if label in values:
            recorded[label] = list(args[1])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Axes, 'plot', capture)
    logs = tmp_path / 'metrics.jsonl'
    logs.write_text(json.dumps(dict(iteration=1, train_mean_steps=100,
        validation=dict(mean_steps=120, mean_return=264,
                        p512=1., p1024=.6, p2048=.2, p4096=.1))) + '\n')
    output = plot_training(logs)
    assert recorded == {key: [value] for key, value in values.items()}
    assert output.exists() and not (tmp_path / 'training.tmp.png').exists()
