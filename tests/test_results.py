"""Published results and new tile thresholds retain their recorded meaning."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from common.evaluation import summarize_results, tile_reach_rate, tile_thresholds
from plot import plot_training, read_metrics


def test_high_tile_rates_are_recovered_from_games_and_extend_automatically(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    games = [dict(seed=i, steps=100, spawn_return=220., board_sum=222,
                  max_tile=tile, merge_score=1000.)
             for i, tile in enumerate((512, 8192, 16384, 32768))]
    metrics = summarize_results(games, 1.)
    assert metrics['p8192'] == .75 and metrics['p16384'] == .5 and metrics['p32768'] == .25
    assert tile_thresholds()[-1] == 16384 and tile_thresholds(32768)[-1] == 32768
    # Existing logs have per-game outcomes but no aggregate rates above 4096.
    for key in ('p8192', 'p16384', 'p32768'):
        del metrics[key]
    captured = {}
    original = Axes.plot
    def capture(self, *args, **kwargs):
        if kwargs.get('label', '').startswith('≥ '):
            captured[kwargs['label']] = list(args[1])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Axes, 'plot', capture)
    log = tmp_path / 'metrics.jsonl'
    log.write_text(json.dumps(dict(iteration=5, train_mean_steps=100, validation=metrics)) + '\n')
    plot_training(log)
    assert captured['≥ 8192'] == [75.] and captured['≥ 16384'] == [50.]
    assert captured['≥ 32768'] == [25.]
    assert np.isnan(tile_reach_rate({'max_tile': 16384}, 8192))
    assert tile_reach_rate({'max_tile': 4096}, 8192) == 0.


def test_published_snapshots_match_logs_and_inference_checkpoints():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / 'assets/results.json').read_text())
    for run in report['runs']:
        metrics = run['metrics']
        assert [g['seed'] for g in metrics['results']] == report['evaluation_seeds']
        assert metrics['mean_steps'] == pytest.approx(np.mean([g['steps'] for g in metrics['results']]))
        for tile in tile_thresholds(metrics['max_tile']):
            assert metrics[f'p{tile}'] == pytest.approx(tile_reach_rate(metrics, tile))
        if run['id'] == 'mcts':
            assert run['search_budget'] == 100
            continue
        log = root / run['log']
        assert hashlib.sha256(log.read_bytes()).hexdigest() == run['snapshot_log_sha256']
        rows = read_metrics(log)
        assert rows[-1]['iteration'] == run['through_iteration']
        selected = next(r for r in rows if r['iteration'] == run['checkpoint_iteration'])
        assert selected['validation'] == metrics
        data = torch.load(root / run['checkpoint'], map_location='cpu', weights_only=True)
        assert data['iteration'] == run['checkpoint_iteration']
        assert data['model_config'] == run['model']
        assert data['reward_objective'] == 'spawn_mass' and data['inference_only']
