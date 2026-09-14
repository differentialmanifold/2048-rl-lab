"""Two-panel training plot: survival and 512/1024/2048/4096 reach rates."""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np


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
              label=f'Training mean ({window} updates)')
    if validations:
        means = [v['mean_steps'] for v in validations]
        left.plot(vi, means, color='#1869a8', linewidth=.8, alpha=.65, label='Fixed-seed validation')
        left.plot(vi, moving_average(means, 5), color='#49338b', linewidth=1.8,
                  label='Validation mean (5 checks)')
        best = int(np.argmax([v.get('mean_return', v.get('mean_score', 0)) for v in validations]))
        left.scatter(vi[best], means[best], color='#d88322', marker='*', s=95, zorder=5,
                     label='Best validation return')
        for tile, color in [(512, '#2878b5'), (1024, '#c58a21'),
                            (2048, '#19734a'), (4096, '#a54732')]:
            right.plot(vi, [v.get(f'p{tile}', np.nan) * 100 for v in validations],
                       color=color, linewidth=1.2, label=f'≥ {tile}')
        right.legend(loc='upper left', fontsize=9, ncol=2)
    else:
        right.text(.5, .5, 'Waiting for first validation', ha='center', va='center', transform=right.transAxes)
    left.set(title='Survival', ylabel='Mean moves per game')
    left.legend(loc='upper left', fontsize=8)
    right.set(title='Tile reach rates · validation', ylabel='Games (%)', ylim=(-3, 103))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', required=True)
    parser.add_argument('--output', help='Defaults to training.png beside metrics.jsonl')
    parser.add_argument('--title', help='Include algorithm and model, e.g. PPO · ResCNN')
    parser.add_argument('--window', type=int, default=50)
    args = parser.parse_args()
    plot_training(args.logs, args.output, args.window, args.title)


if __name__ == '__main__':
    main()
