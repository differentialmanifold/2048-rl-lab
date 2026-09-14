"""A2C: collect fresh games, compute GAE, take one full-batch actor/critic step."""
import numpy as np
import torch
from torch.nn import functional as F

from gym2048_env import Gym2048Env
from common.models import masked_categorical
from common.rollout import collect_actor_critic
from common.evaluation import evaluate
from common.training import TrainingRun, training_parser, resolve_args


def update(model, optimizer, rollout, entropy_coef=0.01, value_coef=0.5):
    # Recompute with gradients: rollout collection intentionally used no_grad.
    logits, values = model(rollout.states)
    distribution = masked_categorical(logits, rollout.masks)
    actor = -(distribution.log_prob(rollout.actions) * rollout.advantages).mean()
    critic = F.smooth_l1_loss(values, rollout.returns)
    entropy = distribution.entropy().mean()
    loss = actor + value_coef * critic - entropy_coef * entropy
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return dict(actor=float(actor.detach()), critic=float(critic.detach()),
                entropy=float(entropy.detach()), updates=1)


def train(args):
    run = TrainingRun(args, 'a2c')
    model, optimizer = run.model, run.optimizer
    env = Gym2048Env()
    run.ensure_baseline(lambda: evaluate(model, args.eval_episodes, args.eval_seed))
    for iteration in range(run.start, args.iterations + 1):
        # 1. Freeze the policy while collecting complete games on fresh seeds.
        rollout = collect_actor_critic(env, model, args.episodes_per_update, args.gamma,
                                      10_000_000 + args.seed + iteration * 100_000, args.gae_lambda)
        # 2. Use this batch once. The next iteration collects new trajectories.
        metrics = update(model, optimizer, rollout, args.entropy_coef, args.value_coef)
        metrics.update(iteration=iteration, transitions=len(rollout.actions),
                       train_mean_return=float(np.mean([e['spawn_return'] for e in rollout.episodes])),
                       train_mean_steps=float(np.mean([e['steps'] for e in rollout.episodes])),
                       train_max_tile=max(e['max_value'] for e in rollout.episodes))
        # 3. Evaluate the updated model without learning, then save/log/plot.
        if run.should_validate(iteration):
            metrics['validation'] = evaluate(model, args.eval_episodes, args.eval_seed)
        run.record(metrics)
    env.close()
    return model


def main(argv=None):
    parser = training_parser(__doc__)
    parser.add_argument('--episodes-per-update', type=int)
    parser.add_argument('--gae-lambda', type=float)
    parser.add_argument('--entropy-coef', type=float)
    parser.add_argument('--value-coef', type=float)
    args = resolve_args(parser, 'a2c', dict(episodes_per_update=8, gae_lambda=.95,
                                          entropy_coef=.01, value_coef=.5), argv)
    return train(args)


if __name__ == '__main__':
    main()
