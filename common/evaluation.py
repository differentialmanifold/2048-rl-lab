"""Fixed-seed evaluation without gradient updates."""
import time
import numpy as np
import torch
from gym2048_env import Gym2048Env
from common.models import REWARD_OBJECTIVE, preprocess_observation, masked_categorical


@torch.no_grad()
def evaluate(model, episodes=10, seed=1_000_000, action_fn=None, pool=None):
    if episodes < 1:
        raise ValueError('Evaluation requires at least one episode')
    if action_fn is None:
        start = time.perf_counter()
        if pool is not None and pool.workers > 1:
            from common.parallel import model_snapshot
            snapshot = model_snapshot(model)
            groups = np.array_split(np.arange(episodes), min(pool.workers, episodes))
            parts = pool.map(_evaluate_worker, [(snapshot, len(ids), seed + int(ids[0])) for ids in groups])
            results = [row for part in parts for row in part]
        else:
            results = _evaluate_batch(model, episodes, seed)
        return summarize_results(results, time.perf_counter() - start)
    was_training = model.training if model is not None else False
    if model is not None:
        model.eval()
    results = []
    start = time.perf_counter()
    try:
        for i in range(episodes):
            env = Gym2048Env()
            state, info = env.reset(seed=seed + i)
            steps = 0
            spawn_return = 0.0
            while True:
                action = int(action_fn(state, info, seed + i))
                if not info['can_move_dir'][action]:
                    raise RuntimeError(f'Evaluator selected illegal action {action}')
                state, reward, done, truncated, info = env.step(action)
                spawn_return += float(reward)
                steps += 1
                if done or truncated:
                    break
            results.append(dict(seed=seed + i, steps=steps, spawn_return=spawn_return,
                                board_sum=int(state.sum()), max_tile=info['max_value'],
                                merge_score=info['merge_score'], legacy_score=info['score']))
            env.close()
    finally:
        if model is not None:
            model.train(was_training)
    return summarize_results(results, time.perf_counter() - start)


def summarize_results(results, seconds):
    episodes, seed = len(results), results[0]['seed']
    tiles = [r['max_tile'] for r in results]
    mean_return = float(np.mean([r['spawn_return'] for r in results]))
    return dict(episodes=episodes, seed=seed, score_metric=REWARD_OBJECTIVE,
                mean_score=mean_return, mean_return=mean_return,
                mean_board_sum=float(np.mean([r['board_sum'] for r in results])),
                mean_merge_score=float(np.mean([r['merge_score'] for r in results])),
                mean_steps=float(np.mean([r['steps'] for r in results])), max_tile=max(tiles),
                **{f'p{tile}': float(np.mean(np.array(tiles) >= tile)) for tile in (512, 1024, 2048, 4096)},
                seconds=seconds, ms_per_move=1000 * seconds / sum(r['steps'] for r in results), results=results)


def _evaluate_worker(job):
    from common.parallel import worker_model
    snapshot, episodes, seed = job
    return _evaluate_batch(worker_model(snapshot), episodes, seed)


@torch.no_grad()
def _evaluate_batch(model, episodes, seed):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    envs = [Gym2048Env() for _ in range(episodes)]
    observations, infos = zip(*[env.reset(seed=seed + i) for i, env in enumerate(envs)])
    observations, infos = list(observations), list(infos)
    totals, steps, results = [0.] * episodes, [0] * episodes, [None] * episodes
    active = list(range(episodes))
    try:
        while active:
            states = torch.stack([preprocess_observation(observations[i]) for i in active]).to(device)
            output = model(states)
            logits = output[0] if isinstance(output, tuple) else output
            dist = masked_categorical(logits, [infos[i]['can_move_dir'] for i in active])
            actions = dist.probs.argmax(-1).cpu().tolist()
            remaining = []
            for i, action in zip(active, actions):
                state, reward, done, truncated, info = envs[i].step(action)
                observations[i], infos[i] = state, info
                totals[i] += reward
                steps[i] += 1
                if done or truncated:
                    results[i] = dict(seed=seed + i, steps=steps[i], spawn_return=totals[i],
                        board_sum=int(state.sum()), max_tile=info['max_value'],
                        merge_score=info['merge_score'], legacy_score=info['score'])
                else:
                    remaining.append(i)
            active = remaining
    finally:
        model.train(was_training)
        for env in envs:
            env.close()
    return results
