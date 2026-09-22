"""Persistent spawn workers for independent games; results retain input order."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import subprocess
import sys
import torch
from common.models import model_from_config


def available_cpu_cores():
    """Respect process affinity; prefer performance/physical cores on macOS."""
    process_count = getattr(os, 'process_cpu_count', os.cpu_count)
    available = process_count() or 1
    try:
        available = min(available, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    if sys.platform == 'darwin':
        for name in ('hw.perflevel0.physicalcpu', 'hw.physicalcpu'):
            try:
                count = int(subprocess.check_output(
                    ['sysctl', '-n', name], stderr=subprocess.DEVNULL, timeout=1, text=True).strip())
                if count > 0:
                    available = min(available, count)
                    break
            except (OSError, subprocess.SubprocessError, ValueError):
                continue
    return max(1, available)


def resolve_workers(workers, games):
    """Automatic CPU budget, bounded by independent games; not a speed guarantee."""
    if games < 1:
        raise ValueError('Games must be positive')
    if workers is not None:
        if workers < 1:
            raise ValueError('Workers must be positive')
        return workers
    # Leave one core for the OS/main process; never create idle game workers.
    return min(games, max(1, available_cpu_cores() - 1))


def _initialize_worker():
    torch.set_num_threads(1)


class GamePool:
    def __init__(self, workers=1):
        if workers < 1:
            raise ValueError('Workers must be positive')
        self.workers = workers
        self.executor = (ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                         initializer=_initialize_worker) if workers > 1 else None)

    def map(self, function, jobs):
        return list(self.executor.map(function, jobs)) if self.executor else list(map(function, jobs))

    def close(self):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def model_snapshot(model):
    # Workers only infer on CPU. The optimizer and accelerator stay in the parent.
    return model.model_config, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def worker_model(snapshot):
    config, weights = snapshot
    # Model initialization must not advance the caller's training RNG in serial mode.
    with torch.random.fork_rng(devices=[]):
        model = model_from_config(config)
    model.load_state_dict(weights)
    return model.eval()
