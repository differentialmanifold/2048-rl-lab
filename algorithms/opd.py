"""On-policy distillation: frozen PPO teacher or the student's AlphaZero search.

Only the student samples environment actions. Each fresh rollout is labelled by
the teacher, then discarded after the update. The default loss follows Thinking
Machines' immediate log-probability reward and importance-sampling estimator.
The four-action exact reverse KL is available as a lower-variance alternative.
"""
from pathlib import Path
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from algorithms.alphazero import MCTS, Node, softmax_visit_probs
from common.checkpoints import read_checkpoint, checkpoint_model_config, restore_checkpoint
from common.evaluation import evaluate
from common.models import masked_categorical
from common.parallel import model_snapshot, worker_model
from common.rollout import collect_actor_critic
from common.training import TrainingRun, training_parser, resolve_args
from gym2048_env import Gym2048Env


def reverse_kl(distribution, teacher_log_probs, masks):
    """Per-state KL(student || teacher), with no gradient into teacher labels."""
    student_logs = distribution.logits.masked_fill(~masks, 0.)
    teacher_logs = teacher_log_probs.detach().masked_fill(~masks, 0.)
    return (distribution.probs * (student_logs - teacher_logs)).sum(-1)


def sampled_distillation_loss(distribution, actions, old_log_probs, teacher_log_probs):
    """Immediate advantage log teacher(a|s) - log behavior(a|s), without GAE.

    Do not normalize this signal or backpropagate through either saved log-prob.
    At the behavior policy its expected gradient equals the exact reverse KL
    gradient. Repeated epochs use the saved behavior denominator, not the teacher.
    """
    old_log_probs = old_log_probs.detach()
    teacher_action_logs = teacher_log_probs.detach().gather(-1, actions[:, None]).squeeze(-1)
    advantage = teacher_action_logs - old_log_probs
    ratio = (distribution.log_prob(actions) - old_log_probs).exp()
    return -(ratio * advantage).mean()


@torch.no_grad()
def ppo_teacher_log_probs(teacher, states, masks, batch_size=256):
    was_training = teacher.training
    teacher.eval()
    try:
        return torch.cat([masked_categorical(teacher(states[ids])[0], masks[ids]).logits
                          for ids in torch.arange(len(states), device=states.device).split(batch_size)])
    finally:
        teacher.train(was_training)


@torch.no_grad()
def _search_labels(model, states, masks, seed, mcts_sims, gamma, c_puct,
                   search_temperature, search_smoothing):
    device = next(model.parameters()).device
    ranks = states.cpu().numpy().astype(np.int64).reshape(-1, 4, 4)
    boards = np.where(ranks > 0, np.left_shift(1, ranks), 0)
    legal_masks = masks.cpu().numpy()
    targets = []
    for index, (board, mask) in enumerate(zip(boards, legal_masks)):
        # An independent random stream per state: partitioning across workers
        # cannot change chance outcomes, student actions or real tile spawns.
        search = MCTS(model, device, c_puct=c_puct, dirichlet_frac=0.,
                      gamma=gamma, seed=seed + index)
        root = Node(1., matrix=board, legal_mask=mask)
        search.run(root, mcts_sims)
        visits = softmax_visit_probs(root.children, search_temperature).astype(np.float64)
        visits[~mask] = 0.
        uniform = mask.astype(np.float64) / mask.sum()
        visits = visits / visits.sum() if visits.sum() else uniform
        # Unvisited legal moves need positive support for finite reverse KL.
        targets.append((1. - search_smoothing) * visits + search_smoothing * uniform)
    probabilities = torch.as_tensor(np.stack(targets), dtype=states.dtype, device=device)
    return probabilities.clamp_min(torch.finfo(states.dtype).tiny).log().masked_fill(
        ~masks.to(device), torch.finfo(states.dtype).min)


def _search_worker(job):
    snapshot, states, masks, seed, options = job
    return _search_labels(worker_model(snapshot), states, masks, seed, **options)


@torch.no_grad()
def search_teacher_log_probs(model, states, masks, seed, *, mcts_sims=100,
                             gamma=.999, c_puct=1.5, search_temperature=1.,
                             search_smoothing=.01, pool=None):
    if mcts_sims < 1 or not math.isfinite(c_puct) or c_puct <= 0:
        raise ValueError('mcts_sims and c_puct must be positive')
    if not math.isfinite(search_temperature) or search_temperature <= 0:
        raise ValueError('search_temperature must be positive')
    if not 0 < search_smoothing < 1:
        raise ValueError('search_smoothing must be in (0, 1)')
    if len(states) < 1 or not masks.any(-1).all():
        raise ValueError('Search labels require nonterminal states')
    options = dict(mcts_sims=mcts_sims, gamma=gamma, c_puct=c_puct,
                   search_temperature=search_temperature, search_smoothing=search_smoothing)
    if pool is not None and pool.workers > 1:
        snapshot = model_snapshot(model)
        groups = np.array_split(np.arange(len(states)), min(pool.workers, len(states)))
        jobs = [(snapshot, states[int(ids[0]):int(ids[-1]) + 1].cpu(),
                 masks[int(ids[0]):int(ids[-1]) + 1].cpu(), seed + int(ids[0]), options)
                for ids in groups]
        return torch.cat(pool.map(_search_worker, jobs)).to(states.device)
    return _search_labels(model, states, masks, seed, **options)


def update(model, optimizer, rollout, teacher_log_probs, *, epochs=4, batch_size=256,
           loss_kind='sampled', value_coef=.5, target_kl=.02):
    if loss_kind not in ('sampled', 'exact'):
        raise ValueError('loss_kind must be sampled or exact')
    teacher_log_probs = teacher_log_probs.detach()
    model.train()
    metrics = dict(actor=0., critic=0., entropy=0., kl=0., reverse_kl=0., updates=0)
    for _ in range(epochs):
        indices = torch.randperm(len(rollout.actions), device=rollout.states.device)
        for ids in indices.split(batch_size):
            logits, values = model(rollout.states[ids])
            masks = rollout.masks[ids]
            distribution = masked_categorical(logits, masks)
            old = torch.distributions.Categorical(logits=rollout.old_logits[ids])
            kl = torch.distributions.kl_divergence(old, distribution).mean()
            metrics['kl'] = float(kl.detach())
            if target_kl > 0 and kl.detach() > target_kl:
                return metrics
            exact_kl = reverse_kl(distribution, teacher_log_probs[ids], masks).mean()
            actor = (exact_kl if loss_kind == 'exact' else sampled_distillation_loss(
                distribution, rollout.actions[ids], rollout.old_log_probs[ids], teacher_log_probs[ids]))
            # Environment rewards only train the value head/shared encoder. They
            # never enter the policy advantage or the distillation reward.
            critic = F.smooth_l1_loss(values, rollout.returns[ids])
            loss = actor + value_coef * critic
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            metrics.update(actor=float(actor.detach()), critic=float(critic.detach()),
                           entropy=float(distribution.entropy().mean().detach()),
                           reverse_kl=float(exact_kl.detach()), updates=metrics['updates'] + 1)
    return metrics


def _validate_actor_checkpoint(data, *, ppo_teacher=False):
    config = checkpoint_model_config(data)
    if config.get('model_type', 'actor_critic') != 'actor_critic':
        raise ValueError('OPD requires a direct ActorCritic checkpoint')
    if ppo_teacher and data['algorithm'] != 'ppo':
        raise ValueError('The PPO teacher must be a PPO checkpoint')
    if not ppo_teacher and config['architecture'] != 'cnn2x2':
        raise ValueError('The OPD student must use cnn2x2')
    if config.get('obs_dim', 16) != 16 or config.get('num_actions', 4) != 4:
        raise ValueError('OPD requires a 4x4 board and four actions')


def _teacher_data(args):
    if args.teacher != 'ppo':
        return None
    if args.resume:
        # Persist the teacher itself: a moved or overwritten source path must
        # never silently change the teacher of a resumed experiment.
        data = read_checkpoint(args.resume).get('distillation_teacher')
        if data is None:
            raise ValueError('PPO OPD resume requires the embedded frozen teacher')
    else:
        source = read_checkpoint(args.teacher_checkpoint)
        data = {key: source[key] for key in ('algorithm', 'model_config', 'model')}
    _validate_actor_checkpoint(data, ppo_teacher=True)
    return data


def train(args):
    teacher_data = _teacher_data(args)
    initial = read_checkpoint(args.init_checkpoint) if args.init_checkpoint else None
    if initial is not None:
        _validate_actor_checkpoint(initial)
    with TrainingRun(args, 'opd') as run:
        model, optimizer = run.model, run.optimizer
        if initial is not None:
            restore_checkpoint(initial, model)
        teacher = None
        extra = {}
        if teacher_data is not None:
            teacher = worker_model((teacher_data['model_config'], teacher_data['model'])).to(run.device)
            teacher.requires_grad_(False)
            extra['distillation_teacher'] = teacher_data
        run.ensure_baseline(lambda: evaluate(model, args.eval_episodes, args.eval_seed, pool=run.pool), extra)
        env = Gym2048Env()
        try:
            for iteration in range(run.start, args.iterations + 1):
                # The model stays fixed through collection AND teacher labelling.
                model.eval()
                started = time.perf_counter()
                rollout = collect_actor_critic(
                    env, model, args.episodes_per_update, args.gamma,
                    10_000_000 + args.seed + iteration * 100_000,
                    args.td_steps, args.td_lambda, pool=run.pool)
                collect_seconds = time.perf_counter() - started
                started = time.perf_counter()
                if args.teacher == 'ppo':
                    labels = ppo_teacher_log_probs(teacher, rollout.states, rollout.masks, args.batch_size)
                else:
                    labels = search_teacher_log_probs(
                        model, rollout.states, rollout.masks,
                        30_000_000 + args.seed + iteration * 100_000,
                        mcts_sims=args.mcts_sims, gamma=args.gamma, c_puct=args.c_puct,
                        search_temperature=args.search_temperature,
                        search_smoothing=args.search_smoothing, pool=run.pool)
                teacher_seconds = time.perf_counter() - started
                sampled_reverse_kl = float((rollout.old_log_probs - labels.gather(
                    -1, rollout.actions[:, None]).squeeze(-1)).mean())
                started = time.perf_counter()
                metrics = update(model, optimizer, rollout, labels, epochs=args.epochs,
                                 batch_size=args.batch_size, loss_kind=args.loss,
                                 value_coef=args.value_coef, target_kl=args.target_kl)
                metrics.update(iteration=iteration, teacher=args.teacher, distillation_loss=args.loss,
                    sampled_reverse_kl=sampled_reverse_kl,
                    collect_seconds=collect_seconds, teacher_seconds=teacher_seconds,
                    update_seconds=time.perf_counter() - started, gamma=args.gamma,
                    td_steps=args.td_steps, td_lambda=args.td_lambda, transitions=len(rollout.actions),
                    train_mean_return=float(np.mean([e['spawn_return'] for e in rollout.episodes])),
                    train_mean_steps=float(np.mean([e['steps'] for e in rollout.episodes])),
                    train_max_tile=max(e['max_value'] for e in rollout.episodes))
                # Select checkpoints by the deployed student's search-free policy.
                if run.should_validate(iteration):
                    metrics['validation'] = evaluate(model, args.eval_episodes, args.eval_seed, pool=run.pool)
                run.record(metrics, extra)
        finally:
            env.close()
        return model


def main(argv=None):
    parser = training_parser(__doc__, architectures=('cnn2x2',))
    parser.add_argument('--teacher', choices=('ppo', 'alphazero'), help='Teacher mode (default: ppo)')
    parser.add_argument('--teacher-checkpoint', help='Frozen PPO teacher; full checkpoint or model export')
    parser.add_argument('--init-checkpoint', help='Optional cnn2x2 student weights for a NEW run')
    parser.add_argument('--loss', choices=('sampled', 'exact'), help='Reverse KL estimator (default: sampled)')
    for flag in ('episodes-per-update', 'epochs', 'batch-size', 'td-steps', 'mcts-sims'):
        parser.add_argument('--' + flag, type=int)
    for flag in ('td-lambda', 'value-coef', 'target-kl', 'c-puct', 'search-temperature', 'search-smoothing'):
        parser.add_argument('--' + flag, type=float)
    args = resolve_args(parser, 'opd', dict(teacher='ppo', teacher_checkpoint=None,
        loss='sampled', episodes_per_update=8, epochs=4, batch_size=256,
        td_steps=10, td_lambda=.5, value_coef=.5, target_kl=.02, mcts_sims=100,
        c_puct=1.5, search_temperature=1., search_smoothing=.01), argv)
    if args.architecture != 'cnn2x2':
        parser.error('The OPD student must use cnn2x2')
    if args.resume and args.init_checkpoint:
        parser.error('--init-checkpoint is only for new runs; use --resume alone to continue')
    if args.teacher == 'ppo' and not args.teacher_checkpoint and not args.resume:
        parser.error('--teacher ppo requires --teacher-checkpoint')
    if args.teacher == 'alphazero' and args.teacher_checkpoint:
        parser.error('AlphaZero uses the current student; omit --teacher-checkpoint')
    for key in ('lr', 'c_puct', 'search_temperature'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'{key.replace("_", "-")} must be finite and positive')
    for key in ('value_coef', 'target_kl'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f'{key.replace("_", "-")} must be finite and nonnegative')
    if not 0 < args.search_smoothing < 1:
        parser.error('search-smoothing must be in (0, 1)')
    if args.teacher == 'alphazero' and args.value_coef == 0:
        parser.error('AlphaZero self-distillation requires a positive value-coef to train search values')
    if args.teacher_checkpoint:
        args.teacher_checkpoint = str(Path(args.teacher_checkpoint).resolve())
    if args.resume:
        previous = read_checkpoint(args.resume)['config']
        if args.teacher != previous['teacher'] or args.teacher_checkpoint != previous['teacher_checkpoint']:
            parser.error('Resume must preserve the teacher; use --init-checkpoint for a new experiment')
    if not args.save_dir and not args.resume:
        args.save_dir = f'checkpoints/opd_{args.teacher}_cnn2x2'
    return train(args)


if __name__ == '__main__':
    main()
