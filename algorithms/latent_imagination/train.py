"""Latent Imagination RL: reinforce a policy inside a frozen neural world.

Run directly, or via `python -m algorithms.latent_imagination.pipeline imagine`.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch

from algorithms.ppo_afterstate import DEFAULTS as PPO_DEFAULTS, KL_METHOD, update
from common.checkpoints import read_checkpoint, model_from_checkpoint, save_checkpoint
from common.evaluation import evaluate
from common.training import TrainingRun, training_parser, resolve_args
from .policy import ImaginationAgent
from .rollout import collect_imagined
from .runtime import isolated_rng
from .world.dynamics import world_fingerprint
from .world.data import atomic_json
from .world.source import verified_source
from .lr_control import KLRateController

# Keep the serialized trainer id so existing runs resume without conversion.
ALGORITHM = 'latent_afterstate_ppo'
DEFAULTS = {**PPO_DEFAULTS, 'architecture': 'vit', 'world_run': '',
            'max_imagined_steps': 10000, 'imagination_device': 'auto',
            'lr_schedule': 'adaptive_kl', 'lr_min': 1e-6, 'lr_max': 3e-4, 'lr_patience': 5}


def assert_frozen(model, fingerprint):
    if any(p.requires_grad or p.grad is not None for p in model.world.parameters()):
        raise RuntimeError('World parameters must stay outside policy optimization')
    if world_fingerprint(model.world) != fingerprint:
        raise RuntimeError('Frozen world weights changed')


def train(args, reset_lr=False):
    # Resume is self-contained: frozen world weights and provenance live in the
    # new checkpoint, so deleting/moving the old run cannot change its dynamics.
    saved = read_checkpoint(args.resume) if args.resume else None
    if saved:
        provenance = saved.get('world_source')
        if not provenance or saved.get('verified_world_sha256') != provenance['world_sha256']:
            raise ValueError('Resume checkpoint lacks verified frozen-world provenance')
        config = saved['model_config']
        factory = lambda: ImaginationAgent(config['world_config'], config['architecture'], config['policy_version'])
    else:
        source, provenance = verified_source(args.world_run)

        def factory():
            model = ImaginationAgent(source['model_config'], args.architecture)
            world = model_from_checkpoint(source).world
            model.world.load_state_dict(world.state_dict())
            return model

    extra = dict(world_source=provenance, verified_world_sha256=provenance['world_sha256'],
                 rollout_source='learned_latent_world', policy_objective='afterstate_ppo',
                 kl_method=KL_METHOD)
    directory = Path(args.save_dir or (Path(args.resume).parent if saved else
                                      f'checkpoints/latent_imagination_{args.architecture}_seed{args.seed}'))
    args.save_dir = str(directory.resolve())
    # Share the world trainer's per-directory lock.
    from .runtime import lock_run
    with lock_run(directory), TrainingRun(args, ALGORITHM, model_factory=factory,
            optimizer_factory=lambda model: torch.optim.Adam(model.policy.parameters(), lr=args.lr)) as run:
        assert_frozen(run.model, provenance['world_sha256'])
        controller_state = saved.get('lr_controller') if saved and not reset_lr else None
        controller = KLRateController(args.lr_schedule, args.lr, args.lr_min, args.lr_max,
                                      args.lr_patience, controller_state)
        controller.apply(run.optimizer)
        extra['lr_controller'] = controller.state_dict()
        atomic_json(directory / 'world_source.json', provenance)
        atomic_json(directory / 'run.json', dict(algorithm=ALGORITHM, config=vars(args),
                    stages=['verified_world', 'imagination'], behavior_required=False,
                    policy_parameters=sum(p.numel() for p in run.model.policy.parameters())))

        def progress(iteration, complete=False):
            atomic_json(directory / 'progress.json', dict(status='complete' if complete else 'training',
                stages=dict(verified_world=dict(status='passed', world_sha256=provenance['world_sha256']),
                            imagination=dict(status='complete' if complete else 'training',
                                iteration=iteration, target=args.iterations, best_real_return=run.best)),
                completion_condition='requested PPO iteration budget; best selected by real evaluation'))

        def real_evaluation():
            with isolated_rng(run.device, args.eval_seed):
                return evaluate(run.model, args.eval_episodes, args.eval_seed, pool=run.pool)

        run.ensure_baseline(real_evaluation, extra)
        if not saved:
            baseline = real_evaluation()
            run.best = baseline['mean_return']
            atomic_json(directory / 'baseline.json', baseline)
            for name in ('last.pt', 'best.pt'):
                save_checkpoint(directory / name, run.model, run.optimizer, 0, ALGORITHM,
                                vars(args), run.best, extra)
            run.logs.write_text(json.dumps(dict(iteration=0, validation=baseline,
                algorithm=ALGORITHM, rollout_source='learned_latent_world')) + '\n')
            from plot import plot_training
            plot_training(run.logs, title='Latent Imagination RL · real validation')
        progress(run.start - 1)
        for iteration in range(run.start, args.iterations + 1):
            started = time.perf_counter()
            rollout = collect_imagined(run.model, args.episodes_per_update, args.gamma,
                10000000 + args.seed + iteration * 100000, args.td_steps, args.td_lambda,
                args.max_imagined_steps, run.pool, args.imagination_device)
            collect_seconds = time.perf_counter() - started
            started = time.perf_counter()
            # Reuse the reference update verbatim: selected-afterstate Huber,
            # PPO clipping, entropy, finite masked-log-prob KL and early stop.
            metrics = update(run.model.policy, run.optimizer, rollout, args.epochs, args.batch_size,
                             args.clip_range, args.entropy_coef, args.value_coef, args.target_kl)
            metrics.update(controller.advance(metrics, args.target_kl))
            controller.apply(run.optimizer)
            extra['lr_controller'] = controller.state_dict()
            assert_frozen(run.model, provenance['world_sha256'])
            metrics.update(iteration=iteration, transitions=len(rollout.actions),
                gamma=args.gamma, td_steps=args.td_steps, td_lambda=args.td_lambda,
                collect_seconds=collect_seconds, update_seconds=time.perf_counter() - started,
                imagined_mean_return=float(np.mean([e['spawn_return'] for e in rollout.episodes])),
                imagined_mean_steps=float(np.mean([e['steps'] for e in rollout.episodes])),
                imagined_truncated_fraction=float(np.mean([e['truncated'] for e in rollout.episodes])),
                imagined_zero_step_games=sum(e['steps'] == 0 for e in rollout.episodes),
                imagined_zero_reward_fraction=sum(e['zero_rewards'] for e in rollout.episodes)/len(rollout.actions),
                imagined_longest_zero_reward_streak=max(e['longest_zero_reward_streak'] for e in rollout.episodes),
                rollout_source='learned_latent_world', world_sha256=provenance['world_sha256'],
                trainable_parameters=sum(p.numel() for p in run.model.policy.parameters()),
                collection_device=('cpu_workers' if args.imagination_device == 'auto' and args.workers > 1
                                   else str(run.device)))
            if run.should_validate(iteration):
                metrics['validation'] = real_evaluation()
            run.record(metrics, extra)
            if iteration % args.plot_every == 0 or iteration == args.iterations:
                from .lr_control import plot_control
                plot_control(run.logs)
            progress(iteration, iteration == args.iterations)
        return run.model


def main(argv=None):
    parser = training_parser(__doc__, architectures=('vit', 'cnn2x2'))
    parser.add_argument('--world-run', help='Completed world pipeline directory; required for a new policy')
    parser.add_argument('--imagination-device', choices=('auto', 'model'),
                        help='auto: CPU workers when workers>1; model: batched imagination on --device')
    parser.add_argument('--lr-schedule', choices=('constant', 'adaptive_kl'),
                        help='New runs default to adaptive_kl; legacy resumes retain constant unless requested')
    parser.add_argument('--lr-min', type=float)
    parser.add_argument('--lr-max', type=float)
    parser.add_argument('--lr-patience', type=int)
    for flag in ('episodes-per-update', 'epochs', 'batch-size', 'td-steps', 'max-imagined-steps'):
        parser.add_argument('--' + flag, type=int)
    for flag in ('td-lambda', 'entropy-coef', 'value-coef', 'clip-range', 'target-kl'):
        parser.add_argument('--' + flag, type=float)
    requested = parser.parse_args(argv)
    args = resolve_args(parser, ALGORITHM, DEFAULTS, argv)
    reset_lr = requested.lr is not None
    if not args.resume and not args.world_run:
        parser.error('A new policy requires --world-run with passed world10 and whole-game audit')
    if args.world_run:
        args.world_run = str(Path(args.world_run).resolve())
    if args.resume:
        previous = read_checkpoint(args.resume)['config']
        if 'lr_schedule' not in previous and requested.lr_schedule is None:
            args.lr_schedule = 'constant'
        if requested.lr_schedule is not None and args.lr_schedule != previous.get('lr_schedule', 'constant'):
            reset_lr = True
        if requested.workers is None:
            args.workers = previous['workers']
            args.workers_auto = previous.get('workers_auto', False)
        if args.world_run != previous['world_run']:
            parser.error('Resume cannot change its verified world; choose a new policy run')
    if args.max_imagined_steps < 1 or not 0 < args.clip_range < 1 or min(args.entropy_coef, args.value_coef, args.target_kl) < 0:
        parser.error('Invalid imagined step limit or PPO loss parameters')
    if not 0 < args.lr_min <= args.lr <= args.lr_max or args.lr_patience < 1:
        parser.error('Require 0 < lr-min <= lr <= lr-max and lr-patience >= 1')
    if args.lr_schedule == 'adaptive_kl' and args.target_kl <= 0:
        parser.error('Adaptive LR requires target-kl > 0')
    args.kl_method = KL_METHOD
    return train(args, reset_lr)


if __name__ == '__main__':
    main()
