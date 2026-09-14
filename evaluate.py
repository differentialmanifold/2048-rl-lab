"""Evaluate a learned policy or search on a separate set of complete games."""
import argparse
import json
from pathlib import Path

from common.evaluation import evaluate
from common.training import setup
from play import make_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=['mcts', 'a2c', 'ppo', 'alphazero'], required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2_000_000)
    parser.add_argument('--budget', type=int, default=500)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', default='experiments/evaluation.json')
    args = parser.parse_args()
    if args.episodes < 1 or args.budget < 1:
        parser.error('Episodes and search budget must be positive')
    device = setup(args.seed, args.device)
    current_seed, controller, metadata = None, None, {}
    def action(state, info, episode_seed):
        nonlocal current_seed, controller, metadata
        if current_seed != episode_seed:
            controller, _, metadata = make_agent(args.agent, args.checkpoint, args.budget, episode_seed, device)
            current_seed = episode_seed
        return controller(state, info)
    results = evaluate(None, args.episodes, args.seed, action)
    results.update(agent=args.agent, **metadata)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
