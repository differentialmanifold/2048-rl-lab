"""Fixed-seed evaluation without gradient updates."""
import time
import numpy as np
import torch
from gym2048_env import Gym2048Env
from common.models import REWARD_OBJECTIVE, preprocess_observation, masked_categorical


@torch.no_grad()
def evaluate(model, episodes=20, seed=1_000_000, action_fn=None):
    if episodes < 1:
        raise ValueError('Evaluation requires at least one episode')
    device = next(model.parameters()).device if model is not None else torch.device('cpu')
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
                if action_fn is None:
                    output = model(preprocess_observation(state).to(device))
                    logits = output[0] if isinstance(output, tuple) else output
                    action = int(masked_categorical(logits, info['can_move_dir']).probs.argmax())
                else:
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
    finally:
        if model is not None:
            model.train(was_training)
    seconds = time.perf_counter() - start
    tiles = [r['max_tile'] for r in results]
    mean_return = float(np.mean([r['spawn_return'] for r in results]))
    return dict(episodes=episodes, seed=seed, score_metric=REWARD_OBJECTIVE,
                mean_score=mean_return, mean_return=mean_return,
                mean_board_sum=float(np.mean([r['board_sum'] for r in results])),
                mean_merge_score=float(np.mean([r['merge_score'] for r in results])),
                mean_steps=float(np.mean([r['steps'] for r in results])), max_tile=max(tiles),
                **{f'p{tile}': float(np.mean(np.array(tiles) >= tile)) for tile in (512, 1024, 2048, 4096)},
                seconds=seconds, ms_per_move=1000 * seconds / sum(r['steps'] for r in results), results=results)
