"""Shared run locks, data identities and isolated evaluation randomness."""
from contextlib import contextmanager
import fcntl
import hashlib

import torch


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()



@contextmanager
def isolated_rng(device, seed):
    cpu = torch.get_rng_state()
    mps = torch.mps.get_rng_state() if device.type == 'mps' else None
    cuda = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    try:
        torch.random.default_generator.manual_seed(seed)
        if mps is not None:
            torch.mps.manual_seed(seed)
        if cuda is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.set_rng_state(cpu)
        if mps is not None:
            torch.mps.set_rng_state(mps)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@contextmanager
def lock_run(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.lock').open('a') as stream:
        try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('Another process is already using this run directory')
        try: yield
        finally: fcntl.flock(stream,fcntl.LOCK_UN)
