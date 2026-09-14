"""CLI, resume and output management. Algorithm training loops live in algorithms/."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from common.models import ActorCritic
from common.checkpoints import read_checkpoint, checkpoint_model_config, restore_checkpoint, save_checkpoint


DEFAULTS = dict(architecture='mlp', seed=0, device='cpu', gamma=1.0, lr=3e-4,
                eval_every=25, eval_episodes=10, eval_seed=1_000_000, plot_every=25)


def setup(seed, device='cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def training_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--iterations', type=int, required=True, help='Total iteration target, including completed iterations on resume')
    parser.add_argument('--architecture', choices=['mlp', 'rescnn'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device')
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--eval-episodes', type=int)
    parser.add_argument('--eval-seed', type=int)
    parser.add_argument('--plot-every', type=int, help='Overwrite training.png every N iterations, and at the end; inherit on resume')
    parser.add_argument('--save-dir')
    parser.add_argument('--resume', help='Resume a full last.pt checkpoint; model-only exports cannot resume')
    return parser


def resolve_args(parser, algorithm, defaults, argv=None):
    args = parser.parse_args(argv)
    settings = {**DEFAULTS, **defaults}
    saved = read_checkpoint(args.resume) if args.resume else None
    if saved:
        if saved['algorithm'] != algorithm:
            parser.error('Checkpoint algorithm does not match this trainer')
        # Original v3 checkpoints used these names; preserve those runs.
        aliases = {'lr': 'learning_rate', 'epochs': 'k_epochs'}
        for key in settings:
            old = saved['config']
            settings[key] = old.get(key, old.get(aliases.get(key), settings[key]))
        settings['architecture'] = checkpoint_model_config(saved)['architecture']
    for key, default in settings.items():
        if getattr(args, key, None) is None:
            setattr(args, key, default)
    if saved:
        for key in ('architecture', 'seed', 'gamma', 'gae_lambda', 'episodes_per_update',
                    'eval_seed', 'eval_episodes', 'eval_mcts_sims'):
            if key in settings and getattr(args, key) != settings[key]:
                parser.error(f'Resume must preserve {key}={settings[key]}')
    if args.iterations < 1 or not 0 <= args.gamma <= 1 or args.lr <= 0:
        parser.error('Iterations and learning rate must be positive; gamma must be in [0, 1]')
    for key in ('eval_every', 'eval_episodes', 'plot_every', 'episodes_per_update', 'epochs',
                'batch_size', 'self_play_games', 'mcts_sims', 'buffer_size', 'train_steps', 'eval_mcts_sims'):
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    return args


class TrainingRun:
    def __init__(self, args, algorithm):
        self.args, self.algorithm = args, algorithm
        self.device = setup(args.seed, args.device)
        self.model = ActorCritic(architecture=args.architecture).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.saved = read_checkpoint(args.resume) if args.resume else None
        self.directory = Path(args.save_dir or (str(Path(args.resume).parent) if args.resume
                              else f'checkpoints/{algorithm}_{args.architecture}'))
        self.logs = self.directory / 'metrics.jsonl'
        if not self.saved and (self.logs.exists() or (self.directory / 'last.pt').exists()):
            raise ValueError(f'{self.directory} already contains training; use --resume or a new --save-dir')
        self.start, self.best = 1, -float('inf')
        if self.saved:
            if self.saved['algorithm'] != algorithm or 'optimizer' not in self.saved:
                raise ValueError('Resume requires a full training checkpoint for this algorithm')
            if self.logs.exists():
                from plot import read_metrics
                if read_metrics(self.logs)[-1]['iteration'] > self.saved['iteration']:
                    raise ValueError('Checkpoint is older than existing logs; use a new --save-dir')
            restore_checkpoint(self.saved, self.model, self.optimizer, restore_rng=True)
            self.start, self.best = self.saved['iteration'] + 1, self.saved['best_metric']
            for group in self.optimizer.param_groups:
                group['lr'] = args.lr
        if self.start > args.iterations:
            raise ValueError(f'Already at iteration {self.start - 1}; increase --iterations')
        self.directory.mkdir(parents=True, exist_ok=True)

    def ensure_baseline(self, evaluator, extra=None):
        """A fork's best model must represent a model actually present in its directory."""
        if self.saved and not (self.directory / 'best.pt').exists():
            metrics = evaluator()
            self.best = metrics['mean_return']
            (self.directory / 'resume_baseline.json').write_text(json.dumps(metrics, indent=2))
            save_checkpoint(self.directory / 'best.pt', self.model, self.optimizer,
                            self.start - 1, self.algorithm, vars(self.args), self.best, extra)

    def should_validate(self, iteration):
        return iteration % self.args.eval_every == 0 or iteration == self.args.iterations

    def record(self, metrics, extra=None):
        """Save each completed iteration, then refresh the plot at its own interval."""
        iteration = metrics['iteration']
        metrics.update(algorithm=self.algorithm, architecture=self.args.architecture)
        improved = False
        if 'validation' in metrics:
            value = metrics['validation']['mean_return']
            improved = value > self.best
            self.best = max(self.best, value)
        final = iteration == self.args.iterations
        if final:
            metrics['stop_reason'] = 'iteration_limit'
        extra = {**(extra or {}), 'stop_reason': 'iteration_limit' if final else None}
        save_checkpoint(self.directory / 'last.pt', self.model, self.optimizer, iteration,
                        self.algorithm, vars(self.args), self.best, extra)
        if improved:
            save_checkpoint(self.directory / 'best.pt', self.model, self.optimizer, iteration,
                            self.algorithm, vars(self.args), self.best, extra)
        with self.logs.open('a') as out:
            out.write(json.dumps(metrics) + '\n')
        print(json.dumps(metrics), flush=True)
        if iteration % self.args.plot_every == 0 or final:
            from plot import plot_training
            plot_training(self.logs, title=f'{self.algorithm.upper()} · {self.args.architecture.upper()}')
