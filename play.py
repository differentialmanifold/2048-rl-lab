"""Play 2048 in a terminal: move yourself, ask the agent, or watch a full game."""
import argparse
from pathlib import Path
import secrets
import sys
import time

import numpy as np
import torch

from gym2048_env import Gym2048Env
from common.models import preprocess_observation, masked_categorical
from common.checkpoints import read_checkpoint, model_from_checkpoint
from common.training import setup

ROOT = Path(__file__).resolve().parent
DIRECTIONS = ['← Left', '↑ Up', '→ Right', '↓ Down']


def resolve_seed(seed=None):
    """Fresh OS randomness for interactive sessions; explicit seeds remain reproducible."""
    return secrets.randbits(32) if seed is None else seed


def make_agent(agent, checkpoint=None, budget=500, seed=None, device='cpu'):
    """Return a state -> action controller plus its visible model provenance."""
    seed = resolve_seed(seed)
    if agent == 'mcts':
        from algorithms.mcts_chance import ChanceMCTS
        planner = ChanceMCTS(budget, seed + 10_000_000)
        return (lambda state, info: planner.search(state)[0], f'MCTS · {budget} rollouts',
                dict(search_budget=budget, transition_model='resampled_chance'))
    path = Path(checkpoint) if checkpoint else ROOT / 'pretrained' / f'{agent}_cnn2x2.pt'
    data = read_checkpoint(path)
    if data['algorithm'] != agent:
        raise ValueError(f'{path} contains {data["algorithm"]}, not {agent}')
    model = model_from_checkpoint(data, device).eval()
    metadata = dict(algorithm=agent, model_config=model.model_config,
                    checkpoint=str(path), checkpoint_iteration=data['iteration'], search_budget=0)
    if agent == 'alphazero':
        from algorithms.alphazero import MCTS, Node, softmax_visit_probs
        gamma = data['config']['gamma']
        planner = MCTS(model, device, dirichlet_frac=0, gamma=gamma, seed=seed)
        def action(state, info):
            root = Node(1., matrix=state.copy(), legal_mask=np.array(info['can_move_dir']))
            planner.run(root, budget)
            return int(softmax_visit_probs(root.children, 0).argmax())
        metadata['search_budget'] = budget
        metadata['reward_objective'] = 'spawn_mass'
    elif agent == 'muzero':
        from algorithms.muzero import search_controller
        action = search_controller(model, budget, data['config'].get('gamma', 1.),
                                   data['config'].get('search_depth', 10), seed)
        metadata.update(search_budget=budget, search_depth=data['config'].get('search_depth', 10))
    else:
        @torch.no_grad()
        def action(state, info):
            logits, _ = model(preprocess_observation(state).to(device))
            return int(masked_categorical(logits, info['can_move_dir']).probs.argmax())
    label = f'{agent.upper()} · {model.architecture.upper()} · iteration {data["iteration"]}'
    return action, label, metadata


def play(agent=None, seed=None, label='Human', auto=False, delay=.1, max_steps=None):
    seed = resolve_seed(seed)
    env = Gym2048Env()
    state, info = env.reset(seed=seed)
    steps, total = 0, 0.
    done, message = False, ''
    try:
        while True:
            if sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            print(f'2048 | {label} | seed={seed}')
            print(env.render())
            print(f'Moves: {steps} | Board sum: {int(state.sum())} | Max: {info["max_value"]} | Merge score: {info["merge_score"]:g}')
            if message:
                print(message)
            if done:
                print('Game over: no legal moves.')
                break
            if max_steps is not None and steps >= max_steps:
                print('Stopped at requested move limit (game may be unfinished).')
                break
            if auto:
                action = agent(state, info)
                if delay:
                    time.sleep(delay)
            else:
                command = input('W/A/S/D + Enter: move | Enter: agent move | H: hint | P: autoplay | Q: quit > ').strip().lower()
                if command == 'q':
                    break
                if command in ('', 'h', 'p') and agent is None:
                    message = 'Choose --agent a2c, ppo, alphazero, muzero, mcts to use an agent.'
                    continue
                if command == 'p':
                    auto = True
                    continue
                if command in ('', 'h'):
                    action = agent(state, info)
                    if command == 'h':
                        message = f'Agent suggests {DIRECTIONS[action]}'
                        continue
                elif command in ('w', 'a', 's', 'd'):
                    action = {'a': 0, 'w': 1, 'd': 2, 's': 3}[command]
                else:
                    message = 'Unknown command.'
                    continue
            if not info['can_move_dir'][action]:
                message = 'That direction cannot move the board.'
                continue
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            total += reward
            message = f'Move {steps}: {DIRECTIONS[action]}'
    except (EOFError, KeyboardInterrupt):
        print('\nGame stopped.')
    finally:
        env.close()
    return dict(steps=steps, spawn_return=total, board_sum=int(state.sum()), max_tile=info['max_value'], terminated=done)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=['human', 'mcts', 'a2c', 'ppo', 'alphazero', 'muzero'], default='human')
    parser.add_argument('--checkpoint', help='Full checkpoint or bundled pretrained/*.pt export')
    parser.add_argument('--budget', type=int, default=500)
    parser.add_argument('--seed', type=int, help='Optional reproducible session; defaults to fresh OS randomness')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--delay', type=float, default=.1)
    parser.add_argument('--max-steps', type=int, help='Optional move limit for a short preview')
    args = parser.parse_args()
    if args.budget < 1 or args.delay < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error('Budget and max-steps must be positive; delay must be nonnegative')
    if args.agent == 'human' and (args.auto or args.checkpoint):
        parser.error('--auto and --checkpoint require an agent')
    args.seed = resolve_seed(args.seed)
    device = setup(args.seed, args.device)
    agent, label = None, 'Human'
    if args.agent != 'human':
        agent, label, _ = make_agent(args.agent, args.checkpoint, args.budget, args.seed, device)
    play(agent, args.seed, label, args.auto, args.delay, args.max_steps)


if __name__ == '__main__':
    main()
