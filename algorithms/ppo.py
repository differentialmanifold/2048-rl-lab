"""PPO with optional D4 policy/value consistency on augmented minibatches."""
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from gym2048_env import Gym2048Env
from common.models import masked_categorical
from common.rollout import collect_actor_critic
from common.evaluation import evaluate
from common.training import TrainingRun, training_parser, resolve_args
from common.symmetry import transform_boards, transform_policy
from common.checkpoints import read_checkpoint


def clipped_surrogate(ratio, advantages, clip_range):
    return torch.minimum(ratio * advantages,
                         ratio.clamp(1 - clip_range, 1 + clip_range) * advantages)


def symmetry_losses(model, states, masks, distribution, values, transforms=None):
    """DrAC-style auxiliary losses, with the necessary D4 action permutation.

    The original predictions are detached teachers. Transformed states NEVER enter
    the PPO importance ratio: their actions were not sampled by the behavior policy.
    Each row draws one of eight transforms; inference remains a single forward pass.
    """
    if transforms is None:
        transforms = torch.randint(8, (len(states),), device=states.device)
    augmented_logits, augmented_values = model(transform_boards(states, transforms))
    augmented_masks = transform_policy(masks, transforms)
    augmented = masked_categorical(augmented_logits, augmented_masks)
    aligned_log_probs = transform_policy(augmented.logits, transforms, inverse=True)
    # Work in log space: legal probabilities may underflow to zero on MPS.
    target_log_probs = distribution.logits.detach().masked_fill(~masks, 0.)
    log_probs = aligned_log_probs.masked_fill(~masks, 0.)
    policy_loss = (distribution.probs.detach() * (target_log_probs - log_probs)).sum(-1).mean()
    value_loss = F.mse_loss(augmented_values, values.detach())
    return policy_loss, value_loss


def update(model, optimizer, rollout, epochs=4, batch_size=256,
           clip_range=.2, entropy_coef=.01, value_coef=.5, target_kl=.02,
           symmetry='none', symmetry_policy_coef=.1, symmetry_value_coef=.1):
    if symmetry not in ('none', 'd4'):
        raise ValueError('symmetry must be none or d4')
    augmented = symmetry == 'd4' and (symmetry_policy_coef > 0 or symmetry_value_coef > 0)
    # Metrics retain the original convention: last successful minibatch, last KL check.
    metrics = dict(actor=0., critic=0., entropy=0., kl=0., updates=0,
                   symmetry=symmetry, symmetry_policy_loss=0., symmetry_value_loss=0.)
    for _ in range(epochs):
        indices = torch.randperm(len(rollout.actions), device=rollout.states.device)
        for ids in indices.split(batch_size):
            logits, values = model(rollout.states[ids])
            distribution = masked_categorical(logits, rollout.masks[ids])
            old_distribution = torch.distributions.Categorical(logits=rollout.old_logits[ids])
            kl = torch.distributions.kl_divergence(old_distribution, distribution).mean()
            metrics['kl'] = float(kl.detach())
            if target_kl > 0 and kl.detach() > target_kl:
                return metrics
            # Sampling and both ratio probabilities use the SAME saved legal mask.
            ratio = (distribution.log_prob(rollout.actions[ids]) - rollout.old_log_probs[ids]).exp()
            actor = -clipped_surrogate(ratio, rollout.advantages[ids], clip_range).mean()
            critic = F.smooth_l1_loss(values, rollout.returns[ids])
            entropy = distribution.entropy().mean()
            loss = actor + value_coef * critic - entropy_coef * entropy
            if augmented:
                policy_symmetry, value_symmetry = symmetry_losses(
                    model, rollout.states[ids], rollout.masks[ids], distribution, values)
                loss = loss + symmetry_policy_coef * policy_symmetry + symmetry_value_coef * value_symmetry
                metrics.update(symmetry_policy_loss=float(policy_symmetry.detach()),
                               symmetry_value_loss=float(value_symmetry.detach()))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            metrics.update(actor=float(actor.detach()), critic=float(critic.detach()),
                           entropy=float(entropy.detach()), updates=metrics['updates'] + 1)
    return metrics


def train(args):
    with TrainingRun(args, 'ppo') as run:
        model, optimizer = run.model, run.optimizer
        env = Gym2048Env()
        run.ensure_baseline(lambda: evaluate(model, args.eval_episodes, args.eval_seed, pool=run.pool))
        for iteration in range(run.start, args.iterations + 1):
            # 1. Collect with a fixed behavior policy and save its masked probabilities.
            started = time.perf_counter()
            rollout = collect_actor_critic(env, model, args.episodes_per_update, args.gamma,
                                          10_000_000 + args.seed + iteration * 100_000, args.td_steps, args.td_lambda, pool=run.pool)
            collect_seconds = time.perf_counter() - started
            started = time.perf_counter()
            # 2. Reuse the batch for a bounded number of epochs; KL may stop it earlier.
            metrics = update(model, optimizer, rollout, args.epochs, args.batch_size,
                             args.clip_range, args.entropy_coef, args.value_coef, args.target_kl,
                             args.symmetry, args.symmetry_policy_coef, args.symmetry_value_coef)
            metrics.update(collect_seconds=collect_seconds, update_seconds=time.perf_counter() - started,
                           gamma=args.gamma, td_steps=args.td_steps, td_lambda=args.td_lambda,
                           iteration=iteration, transitions=len(rollout.actions),
                           train_mean_return=float(np.mean([e['spawn_return'] for e in rollout.episodes])),
                           train_mean_steps=float(np.mean([e['steps'] for e in rollout.episodes])),
                           train_max_tile=max(e['max_value'] for e in rollout.episodes))
            # 3. Evaluate, save and periodically overwrite the two-panel plot.
            if run.should_validate(iteration):
                metrics['validation'] = evaluate(model, args.eval_episodes, args.eval_seed, pool=run.pool)
            run.record(metrics)
        env.close()
        return model


def main(argv=None):
    parser = training_parser(__doc__)
    for flag in ('episodes-per-update', 'epochs', 'batch-size'):
        parser.add_argument('--' + flag, type=int)
    for flag in ('td-lambda', 'entropy-coef', 'value-coef', 'clip-range', 'target-kl'):
        parser.add_argument('--' + flag, type=float)
    parser.add_argument('--symmetry', choices=('none', 'd4'), help='D4 augmentation during training only (default: none)')
    parser.add_argument('--symmetry-policy-coef', type=float, help='Augmented policy KL weight (default: 0.1)')
    parser.add_argument('--symmetry-value-coef', type=float, help='Augmented value MSE weight (default: 0.1)')
    parser.add_argument('--td-steps', type=int)
    args = resolve_args(parser, 'ppo', dict(episodes_per_update=8, epochs=4, batch_size=256,
        td_steps=10, td_lambda=.5, entropy_coef=.01, value_coef=.5, clip_range=.2, target_kl=.02,
        symmetry='none', symmetry_policy_coef=.1, symmetry_value_coef=.1), argv)
    for key in ('symmetry_policy_coef', 'symmetry_value_coef'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f'{key.replace("_", "-")} must be finite and nonnegative')
    if args.resume:
        previous = read_checkpoint(args.resume)['config'].get('symmetry', 'none')
        if previous != args.symmetry and (not args.save_dir or
                Path(args.save_dir).resolve() == Path(args.resume).parent.resolve()):
            parser.error('Changing symmetry on resume requires a new --save-dir for a separate experiment')
    return train(args)


if __name__ == '__main__':
    main()
