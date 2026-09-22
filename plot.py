"""Two-panel training plot: survival and tile reach rates through 16384 and beyond."""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
from common.evaluation import tile_thresholds, tile_reach_rate


def read_metrics(path):
    lines = Path(path).read_text().splitlines()
    rows = {}
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                warnings.warn('Ignoring incomplete final log line (training may still be writing)')
                break
            raise
        if 'iteration' in row:
            rows[int(row['iteration'])] = row
    if not rows:
        raise ValueError(f'No training iterations in {path}')
    return [rows[i] for i in sorted(rows)]


def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError('Smoothing window must be positive')
    finite = np.isfinite(values)
    sums = np.r_[0., np.cumsum(np.where(finite, values, 0.))]
    counts = np.r_[0, np.cumsum(finite)]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(end - window, 0)
    n = counts[end] - counts[start]
    return np.divide(sums[end] - sums[start], n, out=np.full(len(values), np.nan), where=n > 0)


def plot_training(logs, output=None, window=50, title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    logs = Path(logs)
    rows = read_metrics(logs)
    output = Path(output or logs.with_name('training.png'))
    output.parent.mkdir(parents=True, exist_ok=True)
    vi = np.array([r['iteration'] for r in rows if 'validation' in r])
    validations = [r['validation'] for r in rows if 'validation' in r]
    steps = np.array([r.get('train_mean_steps', np.nan) for r in rows])
    updates = np.array([r['iteration'] for r in rows])
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    fig.suptitle(f'{title or logs.parent.name} · through iteration {updates[-1]:,}',
                 fontsize=17, weight='bold')
    left.plot(updates, steps, color='#b9c8d7', linewidth=.5, alpha=.5, label='Training batch')
    left.plot(updates, moving_average(steps, window), color='#657d92', linewidth=1.3,
              label=f'Training mean ({window} iterations)')
    if validations:
        means = [v['mean_steps'] for v in validations]
        left.plot(vi, means, color='#1869a8', linewidth=.8, alpha=.65, label='Fixed-seed validation')
        left.plot(vi, moving_average(means, 5), color='#49338b', linewidth=1.8,
                  label='Validation mean (5 checks)')
        best = int(np.argmax([v.get('mean_return', v.get('mean_score', 0)) for v in validations]))
        left.scatter(vi[best], means[best], color='#d88322', marker='*', s=95, zorder=5,
                     label='Best validation return')
        thresholds = tile_thresholds(max(v.get('max_tile', 0) for v in validations))
        colors = plt.get_cmap('tab10').colors
        for i, tile in enumerate(thresholds):
            rates = [tile_reach_rate(v, tile) * 100 for v in validations]
            color = colors[i % len(colors)]
            right.plot(vi, rates, color=color, linewidth=.5, alpha=.15)
            right.plot(vi, moving_average(rates, 5), color=color, linewidth=1.5, label=f'≥ {tile}')
        right.legend(loc='upper left', fontsize=9, ncol=2)
    else:
        right.text(.5, .5, 'Waiting for first validation', ha='center', va='center', transform=right.transAxes)
    left.set(title='Survival', ylabel='Mean moves per game')
    left.legend(loc='upper left', fontsize=8)
    right.set(title='Tile reach rates · validation (5-check mean)', ylabel='Games (%)', ylim=(-3, 103))
    for ax in (left, right):
        ax.set_xlabel('Training iteration')
        ax.grid(alpha=.16)
    # Readers always see a complete image, including while training is running.
    temporary = output.with_name(output.stem + '.tmp' + output.suffix)
    fig.savefig(temporary, dpi=160)
    temporary.replace(output)
    plt.close(fig)
    print(f'Plot updated: {output}', flush=True)
    return output


def plot_results(results, output=None):
    """One result snapshot per agent: survival bars and a tile-rate heatmap."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = Path(results)
    report = json.loads(results.read_text())
    runs = report['runs']
    output = Path(output or results.with_name('overview.png'))
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [r['label'] + (' †' if r.get('historical_configuration') else '') for r in runs]
    thresholds = tile_thresholds(max(r['metrics']['max_tile'] for r in runs))
    rates = np.array([[100 * tile_reach_rate(r['metrics'], t) for t in thresholds] for r in runs])
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios': [1.1, 1]})
    fig.subplots_adjust(left=.19, right=.98, bottom=.25, top=.80, wspace=.15)
    fig.suptitle('2048 · recorded algorithm results', fontsize=19, weight='bold', y=.97)
    fig.text(.5, .90, f"Snapshot {report['recorded_at']} · 10 games each · seeds 1,000,000–1,000,009",
             ha='center', color='#526171', fontsize=10)
    positions = np.arange(len(runs))
    means = [r['metrics']['mean_steps'] for r in runs]
    bars = left.barh(positions, means, color=['#8695a5', '#438680', '#2b659c', '#7382b9', '#b67c38', '#9b5a70'], height=.62)
    left.bar_label(bars, labels=[f'{v:,.1f}' for v in means], padding=6, fontsize=10)
    left.set(yticks=positions, yticklabels=labels, xlabel='Mean moves per game',
             title='Survival', xlim=(0, max(means) * 1.2))
    left.set_ylim(len(runs) - .5, -.5)
    left.grid(axis='x', alpha=.15)
    left.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        left.spines[side].set_visible(False)
    left.tick_params(axis='y', length=0)
    right.imshow(rates, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    right.set(xticks=np.arange(len(thresholds)), xticklabels=[f'≥{t}' for t in thresholds],
              yticks=[], title='Tile reach rates', xlabel='Highest tile reached in each game')
    right.tick_params(axis='x', labelsize=9, length=0)
    for row in range(len(runs)):
        for col in range(len(thresholds)):
            value = rates[row, col]
            right.text(col, row, f'{value:.0f}%' if np.isfinite(value) else '—',
                       ha='center', va='center', color='white' if value >= 60 else '#20374c', fontsize=10)
    for spine in right.spines.values():
        spine.set_visible(False)
    fig.text(.19, .04, 'MCTS / AlphaZero / MuZero: 100 simulations per move. A2C / PPO: direct policy.\n'
             'Neural results use checkpoint-selection validation games; training budgets differ.\n'
             '† Recorded before the current training defaults. This is a descriptive snapshot, not a controlled ranking.',
             fontsize=9, color='#526171', linespacing=1.5)
    temporary = output.with_name(output.stem + '.tmp' + output.suffix)
    fig.savefig(temporary, dpi=160)
    temporary.replace(output)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--logs', help='Training metrics.jsonl')
    source.add_argument('--results', help='Recorded result snapshots for the overview figure')
    parser.add_argument('--output', help='Defaults to training.png beside metrics.jsonl')
    parser.add_argument('--title', help='Include algorithm and model, e.g. PPO · CNN2×2')
    parser.add_argument('--window', type=int, default=50)
    args = parser.parse_args()
    if args.results:
        plot_results(args.results, args.output)
    else:
        plot_training(args.logs, args.output, args.window, args.title)


if __name__ == '__main__':
    main()
