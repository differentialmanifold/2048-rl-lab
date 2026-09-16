"""PPO: collect games, then optimize the clipped objective on shuffled minibatches."""
import time
import numpy as np
import torch
from torch.nn import functional as F

from gym2048_env import Gym2048Env
from common.models import masked_categorical
from common.rollout import collect_actor_critic
from common.evaluation import evaluate
from common.training import TrainingRun, training_parser, resolve_args


def clipped_surrogate(ratio, advantages, clip_range):
    return torch.minimum(ratio * advantages,
                         ratio.clamp(1 - clip_range, 1 + clip_range) * advantages)


def update(model, optimizer, rollout, epochs=4, batch_size=256,
           clip_range=.2, entropy_coef=.01, value_coef=.5, target_kl=.02):
    # Metrics retain the original convention: last successful minibatch, last KL check.
    metrics = dict(actor=0., critic=0., entropy=0., kl=0., updates=0)
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
                                          10_000_000 + args.seed + iteration * 100_000, args.gae_lambda, pool=run.pool)
            collect_seconds = time.perf_counter() - started
            started = time.perf_counter()
            # 2. Reuse the batch for a bounded number of epochs; KL may stop it earlier.
            metrics = update(model, optimizer, rollout, args.epochs, args.batch_size,
                             args.clip_range, args.entropy_coef, args.value_coef, args.target_kl)
            metrics.update(collect_seconds=collect_seconds, update_seconds=time.perf_counter() - started,
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
    for flag in ('gae-lambda', 'entropy-coef', 'value-coef', 'clip-range', 'target-kl'):
        parser.add_argument('--' + flag, type=float)
    args = resolve_args(parser, 'ppo', dict(episodes_per_update=8, epochs=4, batch_size=256,
        gae_lambda=.95, entropy_coef=.01, value_coef=.5, clip_range=.2, target_kl=.02), argv)
    return train(args)


if __name__ == '__main__':
    main()
