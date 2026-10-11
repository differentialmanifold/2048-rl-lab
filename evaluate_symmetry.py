"""Reproducible PPO D4 audit: held-out games, fixed boards, and coupled continuations."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from board import Board
from common.checkpoints import read_checkpoint, model_from_checkpoint
from common.evaluation import evaluate
from common.models import masked_categorical, preprocess_observation
from common.parallel import GamePool, model_snapshot, worker_model
from common.symmetry import NAMES, transform_boards, transform_policy, transform_actions
from common.training import setup
from gym2048_env import Gym2048Env


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def _source_game(job):
    path, seed, stride, min_tile = job
    model = model_from_checkpoint(read_checkpoint(path)).eval()
    env = Gym2048Env()
    board, info = env.reset(seed=seed)
    candidates, step = [], 0
    try:
        while True:
            if step % stride == 0 and board.max() >= min_tile and sum(info['can_move_dir']) >= 2:
                candidates.append(dict(source=str(path), seed=seed, step=step, board=board.tolist()))
            logits, _ = model(preprocess_observation(board))
            action = int(masked_categorical(logits, info['can_move_dir']).probs.argmax())
            board, _, done, truncated, info = env.step(action)
            step += 1
            if done or truncated:
                return candidates
    finally:
        env.close()


def make_corpus(paths, episodes, per_source, seed, workers, stride=32, min_tile=512):
    """Stratify real reachable midgames by largest tile and empty-cell count."""
    rng = np.random.default_rng(seed)
    selected, seen = [], set()
    with GamePool(workers) as pool:
        for source_index, path in enumerate(paths):
            games = pool.map(_source_game, [(path, seed + source_index * 10000 + i, stride, min_tile)
                                            for i in range(episodes)])
            buckets = defaultdict(list)
            for row in itertools.chain.from_iterable(games):
                board = np.asarray(row['board'])
                buckets[(int(board.max()), int((board == 0).sum()) // 2)].append(row)
            for bucket in buckets.values():
                rng.shuffle(bucket)
            rows = []
            while buckets and len(rows) < per_source:
                for key in sorted(list(buckets)):
                    row = buckets[key].pop()
                    if not buckets[key]:
                        del buckets[key]
                    flat = torch.tensor(row['board']).reshape(16)
                    canonical = min(transform_boards(flat, g).numpy().tobytes() for g in range(8))
                    if canonical not in seen:
                        rows.append(row)
                        seen.add(canonical)
                    if len(rows) == per_source:
                        break
            if len(rows) < per_source:
                raise ValueError(f'{path}: only {len(rows)} distinct midgames; increase --source-episodes')
            selected.extend(rows)
    return dict(format='2048_d4_audit_v1', seed=seed, source_episodes=episodes,
                boards_per_source=per_source, stride=stride, min_tile=min_tile,
                sources=[dict(path=str(p), sha256=fingerprint(p)) for p in paths], boards=selected)


@torch.no_grad()
def policy_diagnostics(model, boards, batch_size=256):
    """Compare all 28 pairs of aligned policies; no augmented inference ensemble."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    probabilities, predictions, choices = [], [], []
    try:
        for start in range(0, len(boards), batch_size):
            states = torch.stack([preprocess_observation(b) for b in boards[start:start + batch_size]]).to(device)
            masks = torch.tensor([Board(b).can_move_dir for b in boards[start:start + batch_size]], device=device)
            if not masks.any(-1).all():
                raise ValueError('Symmetry diagnostics require nonterminal boards')
            aligned, values, greedy_actions = [], [], []
            for g in range(8):
                logits, value = model(transform_boards(states, g))
                policy = masked_categorical(logits, transform_policy(masks, g))
                aligned.append(transform_policy(policy.probs, g, inverse=True).cpu())
                values.append(value.cpu())
                greedy_actions.append(transform_actions(policy.probs.argmax(-1), g, inverse=True).cpu())
            probabilities.append(torch.stack(aligned, dim=1))
            predictions.append(torch.stack(values, dim=1))
            choices.append(torch.stack(greedy_actions, dim=1))
    finally:
        model.train(was_training)
    probs = torch.cat(probabilities).double().numpy()
    values = torch.cat(predictions).double().numpy()
    tv = np.stack([np.abs(probs[:, a] - probs[:, b]).sum(-1) / 2
                   for a, b in itertools.combinations(range(8), 2)], axis=1)
    greedy = torch.cat(choices).numpy()
    agreement = np.stack([greedy[:, a] == greedy[:, b]
                          for a, b in itertools.combinations(range(8), 2)], axis=1)
    best_sets = np.isclose(probs, probs.max(-1, keepdims=True), atol=1e-6, rtol=0)
    spread = np.ptp(values, axis=1)
    return dict(boards=len(boards), comparisons_per_board=28,
                mean_pairwise_policy_tv=float(tv.mean()), p95_pairwise_policy_tv=float(np.quantile(tv, .95)),
                max_pairwise_policy_tv=float(tv.max()),
                pairwise_greedy_agreement=float(agreement.mean()),
                all_eight_greedy_agreement=float((greedy == greedy[:, :1]).all(1).mean()),
                all_eight_argmax_set_agreement=float((best_sets == best_sets[:, :1]).all((1, 2)).mean()),
                mean_value_spread=float(spread.mean()), max_value_spread=float(spread.max()))


@torch.no_grad()
def _continue_boards(job):
    """Keep envs in original coordinates and rotate the policy's view.

Identical seeds now couple the *physical* spawn streams across all eight views.
Using the same seed in eight rotated envs would not couple spawn coordinates.
No time limit: report true remaining return, without horizon censoring.
    """
    snapshot, indexed_boards, seed = job
    model = worker_model(snapshot)
    envs, observations, masks, transforms, keys = [], [], [], [], []
    for index, board in indexed_boards:
        for g in range(8):
            env = Gym2048Env()
            observation, info = env.reset_to_matrix(board, seed=seed + index)
            envs.append(env)
            observations.append(observation)
            masks.append(info['can_move_dir'])
            transforms.append(g)
            keys.append((index, g))
    totals, steps = np.zeros(len(envs)), np.zeros(len(envs), dtype=int)
    results, active = [None] * len(envs), list(range(len(envs)))
    try:
        while active:
            states = torch.stack([preprocess_observation(observations[i]) for i in active])
            g = torch.tensor([transforms[i] for i in active])
            mask = torch.tensor([masks[i] for i in active])
            logits, _ = model(transform_boards(states, g))
            chosen = masked_categorical(logits, transform_policy(mask, g)).probs.argmax(-1)
            actions = transform_actions(chosen, g, inverse=True).tolist()
            remaining = []
            for i, action in zip(active, actions):
                observation, reward, done, truncated, info = envs[i].step(action)
                totals[i] += reward
                steps[i] += 1
                observations[i], masks[i] = observation, info['can_move_dir']
                if done or truncated:
                    index, orientation = keys[i]
                    results[i] = dict(board_index=index, transform=NAMES[orientation],
                        seed=seed + index, spawn_return=float(totals[i]), steps=int(steps[i]),
                        max_tile=info['max_value'], merge_score=info['merge_score'])
                else:
                    remaining.append(i)
            active = remaining
        return results
    finally:
        for env in envs:
            env.close()


def continuations(model, boards, seed, pool):
    snapshot = model_snapshot(model)
    # At most 32 simultaneous games per worker; one board's orbit stays together.
    indexed = list(enumerate(boards))
    parts = pool.map(_continue_boards, [(snapshot, indexed[i:i + 4], seed)
                                       for i in range(0, len(indexed), 4)])
    results = [r for part in parts for r in part]
    returns = np.array([r['spawn_return'] for r in results]).reshape(-1, 8)
    return dict(boards=len(boards), games=len(results), spawn_seed=seed,
                mean_return=float(returns.mean()), mean_worst_orientation_return=float(returns.min(1).mean()),
                mean_orientation_spread=float(np.ptp(returns, axis=1).mean()),
                mean_orientation_cv=float((returns.std(1) / np.maximum(returns.mean(1), 1)).mean()),
                mean_steps=float(np.mean([r['steps'] for r in results])), results=results)


def paired_difference(before, after, seed=918273):
    """Percentile bootstrap of paired games, or board-level orbit means."""
    differences = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(1)
    return dict(pairs=len(differences), mean_change=float(differences.mean()),
                ci95=np.quantile(samples, [.025, .975]).tolist(),
                percent_change=float(100 * differences.mean() / max(float(np.mean(before)), 1.)))


def require_completed_run(path, checkpoint, iterations):
    """A best checkpoint may be early; its training run must have finished first.

    Published inference-only baselines have no training directory and are exempt.
    """
    if iterations is None or checkpoint.get('inference_only', False):
        return
    final_path = Path(path).parent / 'last.pt'
    if not final_path.exists():
        raise ValueError(f'{path}: missing last.pt; cannot verify training completion')
    final = read_checkpoint(final_path)
    if final['iteration'] < iterations or final.get('stop_reason') != 'iteration_limit':
        raise ValueError(f'{path}: training has not completed at least {iterations} iterations '
                         f'(last.pt is at {final["iteration"]}, stop_reason={final.get("stop_reason")})')
    if final['model_config'] != checkpoint['model_config'] or final['algorithm'] != checkpoint['algorithm']:
        raise ValueError(f'{path}: last.pt belongs to a different model/algorithm')


def compare(args):
    corpus = json.loads(Path(args.corpus).read_text())
    all_boards = [r['board'] for r in corpus['boards']]
    # Spread continuation cases across the entire stratified corpus, including every source.
    selected = np.linspace(0, len(all_boards) - 1, min(args.continuation_boards, len(all_boards)), dtype=int)
    report = dict(format='2048_d4_comparison_v1', corpus=str(args.corpus),
                  corpus_sha256=fingerprint(args.corpus), continuation_board_indices=selected.tolist(),
                  inference='single_forward_greedy', completed_iterations=args.completed_iterations,
                  runs=[], comparisons=[])
    checkpoints = []
    for specification in args.checkpoints:
        label, path = specification.split('=', 1)
        checkpoint = read_checkpoint(path)
        require_completed_run(path, checkpoint, args.completed_iterations)
        checkpoints.append((label, path, checkpoint))
    with GamePool(args.workers) as pool:
        for label, path, checkpoint in checkpoints:
            model = model_from_checkpoint(checkpoint).eval()
            started = time.perf_counter()
            print(json.dumps(dict(event='evaluate', label=label, checkpoint=path)), flush=True)
            result = dict(label=label, checkpoint=path, checkpoint_sha256=fingerprint(path),
                iteration=checkpoint['iteration'], config=checkpoint['config'],
                games=evaluate(model, args.episodes, args.seed, pool=pool),
                symmetry=policy_diagnostics(model, all_boards))
            if len(selected):
                result['continuations'] = continuations(model, [all_boards[i] for i in selected],
                                                        args.seed + 1_000_000, pool)
            result['seconds'] = time.perf_counter() - started
            report['runs'].append(result)
            for previous in report['runs'][:-1]:
                comparison = dict(before=previous['label'], after=label,
                    games=paired_difference([r['spawn_return'] for r in previous['games']['results']],
                                            [r['spawn_return'] for r in result['games']['results']]))
                if len(selected):
                    a = np.array([r['spawn_return'] for r in previous['continuations']['results']]).reshape(-1, 8).mean(1)
                    b = np.array([r['spawn_return'] for r in result['continuations']['results']]).reshape(-1, 8).mean(1)
                    comparison['continuations'] = paired_difference(a, b)
                report['comparisons'].append(comparison)
            write_json(args.output, report)
            print(json.dumps(dict(event='evaluated', label=label, mean_return=result['games']['mean_return'],
                                  symmetry=result['symmetry'], seconds=result['seconds'])), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    corpus = sub.add_parser('corpus', help='Freeze held-out midgames from existing actor-critic policies')
    corpus.add_argument('--source-checkpoints', nargs='+', required=True)
    corpus.add_argument('--source-episodes', type=int, default=4)
    corpus.add_argument('--boards-per-source', type=int, default=64)
    corpus.add_argument('--seed', type=int, default=3_000_000)
    corpus.add_argument('--workers', type=int, default=4)
    corpus.add_argument('--output', required=True)
    audit = sub.add_parser('compare', help='Compare checkpoints on the same games and D4 orbits')
    audit.add_argument('--checkpoints', nargs='+', required=True, metavar='LABEL=PATH')
    audit.add_argument('--corpus', required=True)
    audit.add_argument('--episodes', type=int, default=100)
    audit.add_argument('--continuation-boards', type=int, default=16)
    audit.add_argument('--seed', type=int, default=2_000_000)
    audit.add_argument('--completed-iterations', type=int,
                       help='Require training last.pt to have finished at least N iterations; published inference-only baselines are exempt')
    audit.add_argument('--workers', type=int, default=4)
    audit.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    for key in ('workers', 'episodes', 'source_episodes', 'boards_per_source'):
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    if args.command == 'compare':
        if args.continuation_boards < 0 or any('=' not in p for p in args.checkpoints):
            parser.error('Use nonnegative --continuation-boards and LABEL=PATH checkpoints')
        if args.completed_iterations is not None and args.completed_iterations < 1:
            parser.error('completed-iterations must be positive')
    setup(args.seed, 'cpu')
    if args.command == 'corpus':
        if Path(args.output).exists():
            parser.error('Corpus already exists; reuse it or choose a new output path')
        data = make_corpus(args.source_checkpoints, args.source_episodes, args.boards_per_source,
                           args.seed, args.workers)
        write_json(args.output, data)
        print(json.dumps(dict(output=args.output, boards=len(data['boards']))), flush=True)
    else:
        compare(args)


if __name__ == '__main__':
    main()
