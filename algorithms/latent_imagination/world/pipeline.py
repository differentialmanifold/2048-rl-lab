"""Construct a world with base fitting, exploration and long-trajectory fitting."""
import argparse
import json
from pathlib import Path

from common.checkpoints import read_checkpoint
from ..runtime import lock_run
from .data import atomic_json
from .source import verified_source
from . import base, trajectories

DEFAULTS = dict(device='auto', workers=8, seed=0, base_world='', policy_checkpoint='',
    bootstrap_iterations=10000, trajectory_updates=2000, max_trajectory_updates=0,
    episodes=64, horizon=10, eval_every=200, plot_every=25, refresh_every=1000,
    families=6000, head_families=12000, max_rank=12, width=128, latent_dim=16,
    layers=2, heads=4, max_updates_per_stage=0)
RUNTIME = {'device', 'workers', 'trajectory_updates', 'max_trajectory_updates',
           'max_updates_per_stage'}
STAGES = ('base', 'exploration', 'trajectories')


def configuration(args, root):
    path = root / 'world_pipeline.json'
    old = json.loads(path.read_text()) if path.exists() else None
    if old and old.get('version') != 1:
        raise ValueError('Unsupported world pipeline version')
    cfg = {**DEFAULTS, **(old['config'] if old else {})}
    for key in DEFAULTS:
        value = getattr(args, key)
        if value is None:
            continue
        if key in ('base_world', 'policy_checkpoint') and value:
            value = str(Path(value).resolve())
        if old and key not in RUNTIME and value != cfg[key]:
            raise ValueError(f'World resume must preserve {key}={cfg[key]}')
        if old and key == 'trajectory_updates' and value < cfg[key]:
            raise ValueError('Trajectory update target cannot decrease')
        cfg[key] = value
    for key, value in cfg.items():
        if isinstance(value, int) and value < (0 if key in ('seed', 'max_trajectory_updates',
                                                           'max_updates_per_stage') else 1):
            raise ValueError(f'Invalid world option: {key}')
    if cfg['horizon'] != 10 or cfg['episodes'] < 4:
        raise ValueError('World construction requires horizon=10 and episodes>=4')
    for key in ('base_world', 'policy_checkpoint'):
        if cfg[key]:
            cfg[key] = str(Path(cfg[key]).resolve())
    external = Path(cfg['base_world']) if cfg['base_world'] else None
    if external is not None and (external == root or root in external.parents):
        raise ValueError('External base world must be outside the construction directory')
    state = old or dict(version=1, status='running', stages={})
    state['config'] = cfg
    atomic_json(path, state)
    return cfg, state


def build(root, cfg, state):
    """A failed stage stops its successors; repeating the command resumes."""
    def status(stage, value):
        state['stages'][stage] = dict(status=value)
        state['status'] = value if value in ('paused', 'failed') else 'running'
        atomic_json(root / 'world_pipeline.json', state)

    base_root = Path(cfg['base_world']) if cfg['base_world'] else root / 'base'
    base_done = state['stages'].get('base', {}).get('status') == 'passed'
    status('base', 'training')
    if not cfg['base_world'] and not base_done:
        options = ['run', '--run-dir', str(base_root)]
        for key in ('device', 'workers', 'seed', 'families', 'head_families', 'max_rank',
                    'width', 'latent_dim', 'layers', 'heads', 'max_updates_per_stage'):
            options += ['--' + key.replace('_', '-'), str(cfg[key])]
        try:
            base.main(options)
        except SystemExit as error:
            if error.code != 2:
                raise
            status('base', 'paused')
            return False
    # Accept the checkpoint and reports even when a completed stage is reused.
    _, provenance = verified_source(base_root)
    base_root = Path(provenance['world_run'])
    status('base', 'passed')

    status('exploration', 'training')
    policy = Path(cfg['policy_checkpoint']) if cfg['policy_checkpoint'] else root / 'exploration/last.pt'
    if not cfg['policy_checkpoint']:
        iteration = read_checkpoint(policy)['iteration'] if policy.exists() else -1
        if iteration < cfg['bootstrap_iterations']:
            from ..train import main as train
            options = ['--iterations', str(cfg['bootstrap_iterations']), '--device', cfg['device'],
                       '--workers', str(cfg['workers']), '--seed', str(cfg['seed'])]
            if policy.exists():
                options += ['--resume', str(policy)]
            else:
                options += ['--world-run', str(base_root), '--save-dir', str(policy.parent),
                            '--architecture', 'vit', '--eval-episodes', '20']
            train(options)
    teacher = read_checkpoint(policy)
    if (teacher.get('algorithm') != 'latent_afterstate_ppo'
            or teacher.get('verified_world_sha256') != provenance['world_sha256']):
        status('exploration', 'failed')
        raise ValueError('Exploration policy must use this base world')
    status('exploration', 'passed')

    status('trajectories', 'training')
    options = ['--run-dir', str(root / 'world'), '--world-run', str(base_root),
               '--policy-checkpoint', str(policy), '--updates', str(cfg['trajectory_updates']),
               '--max-updates', str(cfg['max_trajectory_updates'])]
    for key in ('device', 'workers', 'seed', 'episodes', 'horizon', 'eval_every',
                'plot_every', 'refresh_every'):
        options += ['--' + key.replace('_', '-'), str(cfg[key])]
    if trajectories.main(options) is False:
        status('trajectories', 'paused')
        return False
    _, accepted = verified_source(root / 'world')
    status('trajectories', 'passed')
    state.update(status='complete', world_sha256=accepted['world_sha256'], world_run=str(root / 'world'))
    atomic_json(root / 'world_pipeline.json', state)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--status', action='store_true', help='Read construction progress')
    parser.add_argument('--base-world', help='Reuse an accepted base world instead of fitting from scratch')
    parser.add_argument('--policy-checkpoint', help='Reuse an exploration policy from that base world')
    for key, value in DEFAULTS.items():
        if key not in ('base_world', 'policy_checkpoint'):
            parser.add_argument('--' + key.replace('_', '-'), type=type(value))
    args = parser.parse_args(argv)
    root = Path(args.run_dir).resolve()
    if args.status:
        print((root / 'world_pipeline.json').read_text())
        return True
    with lock_run(root):
        cfg, state = configuration(args, root)
        return build(root, cfg, state)
