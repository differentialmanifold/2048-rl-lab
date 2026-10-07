"""Load the frozen world whose checkpoint matches its acceptance reports."""
import json
from pathlib import Path

from common.checkpoints import read_checkpoint, model_from_checkpoint
from ..runtime import file_digest
from .data import SyntheticData
from .boundaries import CurriculumData
from .dynamics import world_fingerprint


def require_world(root,world_data):
    """Load a world only after stage and whole-game checks pass."""
    checkpoint=root/'world10'/'passed.pt'
    short=root/'world10'/'validation.json';whole=root/'audit'/'validation.json'
    if not all(p.exists() for p in (checkpoint,short,whole)):
        raise ValueError('World stage and whole-game audit must pass before imagination training')
    first=json.loads(short.read_text());report=json.loads(whole.read_text())
    digest=file_digest(checkpoint)
    if (first.get('passed') is not True or first.get('consecutive_passes',0)<2
            or report.get('passed') is not True or report.get('split')!='validation'
            or not report.get('games') or not all(g.get('passed') is True for g in report['games'])
            or any(r.get('data_sha256')!=world_data.digest or r.get('checkpoint_sha256')!=digest
                   for r in (first,report))):
        raise ValueError('Passing world/audit reports do not match this checkpoint and dataset')
    frozen=world_fingerprint(model_from_checkpoint(read_checkpoint(checkpoint)).world)
    if report.get('world_sha256')!=frozen:
        raise ValueError('Audited world weights changed')
    return checkpoint,report



def verified_source(world_run):
    root = Path(world_run).resolve()
    pipeline = None
    manifest = root / 'world_pipeline.json'
    if manifest.exists():
        pipeline = json.loads(manifest.read_text())
        if (pipeline.get('version') != 1 or pipeline.get('status') != 'complete'
                or any(pipeline.get('stages', {}).get(stage, {}).get('status') != 'passed'
                       for stage in ('base', 'exploration', 'trajectories'))):
            raise ValueError('World construction is incomplete; finish pipeline world first')
        root = root / 'world'

    data = CurriculumData(SyntheticData(root / 'data'), root / 'head_data')
    checkpoint, report = require_world(root, data)
    source = read_checkpoint(checkpoint)
    if source['model_config'].get('model_version') != 2:
        raise ValueError('A verified neural v2 world is required')
    from .trajectories import require_trajectory_report
    trajectory_report = require_trajectory_report(root, source)
    provenance = dict(world_run=str(root), checkpoint=str(checkpoint),
                      checkpoint_sha256=file_digest(checkpoint), world_sha256=report['world_sha256'],
                      world_dataset_sha256=data.digest, world_iteration=source['iteration'],
                      world_validation=json.loads((root / 'world10/validation.json').read_text()),
                      world_audit=report)
    if pipeline and pipeline.get('world_sha256') != report['world_sha256']:
        raise ValueError('Constructed world identity differs from its pipeline manifest')
    if trajectory_report is not None:
        provenance['trajectory_training'] = trajectory_report
    return source, provenance
