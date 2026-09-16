"""Evaluate a learned policy or search on a separate set of complete games."""
import argparse
import json
import time
from pathlib import Path

from common.evaluation import evaluate, summarize_results
from common.parallel import GamePool, resolve_workers
from common.training import setup
from play import make_agent


def evaluate_game(job):
    agent, checkpoint, budget, seed, device = job
    controller, _, metadata = make_agent(agent, checkpoint, budget, seed, device)
    result = evaluate(None, 1, seed, lambda state, info, _: controller(state, info))
    return result['results'][0], metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=['mcts', 'a2c', 'ppo', 'alphazero'], required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2_000_000)
    parser.add_argument('--budget', type=int, default=500)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--workers', type=int, help='Override automatic CPU/game-based worker count')
    parser.add_argument('--output', default='experiments/evaluation.json')
    args = parser.parse_args()
    if args.episodes < 1 or args.budget < 1:
        parser.error('Episodes, workers and search budget must be positive')
    try:
        args.workers = resolve_workers(args.workers, args.episodes)
    except ValueError as error:
        parser.error(str(error))
    device = setup(args.seed, args.device)
    worker_device = 'cpu' if args.workers > 1 else str(device)
    start = time.perf_counter()
    with GamePool(args.workers) as pool:
        games = pool.map(evaluate_game, [(args.agent, args.checkpoint, args.budget,
                                         args.seed + i, worker_device) for i in range(args.episodes)])
    results = summarize_results([row for row, _ in games], time.perf_counter() - start)
    metadata = games[0][1]
    results.update(agent=args.agent, workers=args.workers, **metadata)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
