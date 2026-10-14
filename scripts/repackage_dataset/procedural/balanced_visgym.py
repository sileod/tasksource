"""Build one balanced VisGym mirror from reproducible, verified family checkpoints."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

from .common import file_sha256
from .scale import collect, examples
from .suites import VISGYM
from ..vision import _materialize

FAMILIES = {'maze':'visgym-mcq', 'equation':'visgym-equation-mcq',
            'rotation':'visgym-rotation-mcq', 'blocks':'visgym-block-mcq'}


def family_selection(root, train, evaluation):
    directories = sorted(root.glob('chunks/*/build/*-release'))
    groups = {'validation':{}, 'train':{}}
    fingerprints = {split:set() for split in groups}; excluded = Counter()
    used = []
    for directory in directories:
        report = json.loads((directory/'provenance.json').read_text())
        if report['revision'] != VISGYM[1]:raise ValueError('Unexpected VisGym source revision')
        for name,digest in report['parquet_sha256'].items():
            if file_sha256(directory/name) != digest:raise ValueError('Checkpoint checksum mismatch')
        collect(directory,groups,fingerprints,train,excluded); used.append(directory)
        if len(fingerprints['train']) >= train:break
    # Keep held-out seeds and deterministic source order, then trim the held-out cap.
    groups['validation'] = {key:qas[:evaluation] for key,qas in groups['validation'].items()}
    heldout = {}; remaining = evaluation
    for key,qas in groups['validation'].items():
        if not remaining:break
        heldout[key] = qas[:remaining]; remaining -= len(heldout[key])
    groups['validation'] = heldout
    if len(fingerprints['train']) != train or remaining:
        raise ValueError('Insufficient verified family checkpoints')
    return used,groups,excluded


def visgym_balanced(max_rows=100000, max_rows_eval=100, seed=42, checkpoints=None):
    """Balanced static VisGym: mazes, equation repair, stick rotation and block legality.

    One mirror, with 25,000 training and 25 held-out examples per family at the
    default limits. Every choice is verified against its instance. Fresh native
    procedural images only; no photographic assets or published benchmark rows.
    Metadata records family, seed, geometry and verification. Apache-2.0.
    """
    if min(max_rows,max_rows_eval)<4 or max_rows%4 or max_rows_eval%4:
        raise ValueError('VisGym limits must be positive multiples of four')
    train,evaluation = max_rows//4,max_rows_eval//4
    supplied = json.loads(Path(checkpoints).read_text()) if checkpoints else {}
    selections = {}; reports = {}; excluded = Counter()
    for family,name in FAMILIES.items():
        root = Path(supplied.get(family,Path('build/visgym-checkpoints')/family)).resolve()
        if family not in supplied:
            subprocess.run([sys.executable,'-m','scripts.repackage_dataset.procedural.scale',name,
                '--output',str(root),'--train',str(train),'--eval',str(evaluation*4),'--seed',str(seed),
                '--candidate-multiplier','4'],check=True)
        used,groups,dropped = family_selection(root,train,evaluation)
        selections[family] = (used,groups); excluded.update(dropped)
        reports[family] = [json.loads((p/'provenance.json').read_text()) for p in used]
    heldout = {key for _,groups in selections.values() for key in groups['validation']}
    training = [key for _,groups in selections.values() for key in groups['train']]
    if heldout.intersection(training) or len(training)!=len(set(training)):
        raise ValueError('Image groups overlap across families or splits')
    def rows(split):
        for family,(directories,groups) in selections.items():
            for row in examples(directories,split,groups[split]):
                for qa in row['qa']:
                    qa['metadata'] = json.dumps({**json.loads(qa['metadata']),'family':family},sort_keys=True)
                yield row
    first = next(iter(selections.values()))[0][0]/'train.parquet'
    features = Dataset.from_parquet(str(first)).features
    streams = IterableDatasetDict({s:IterableDataset.from_generator(rows,gen_kwargs={'split':s},features=features)
                                   for s in ('train','validation')})
    repo = 'visgym-mcq'; source = 'https://github.com/'+VISGYM[0]
    dataset = _materialize(streams,repo,source,VISGYM[1],[],visgym_balanced,'apache-2.0',source+'/blob/'+VISGYM[1]+'/LICENSE')
    dataset = DatasetDict({s:ds.shuffle(seed=seed) for s,ds in dataset.items()})
    folder = Path('build/visgym-mcq-release')
    for split,ds in dataset.items():ds.to_parquet(str(folder/(split+'.parquet')))
    path = folder/'provenance.json'; report = json.loads(path.read_text())
    report.update(mirror='tasksource/visgym-mcq',annotation_origin='procedural',configuration=dict(seed=seed,train_per_family=train,
        evaluation_per_family=evaluation,families=list(FAMILIES),shuffle=True),
        family_counts={f:dict(train=train,validation=evaluation) for f in FAMILIES},
        checkpoint_provenance=reports,selection_exclusions=dict(excluded),
        scripts_sha256={Path(__file__).name:file_sha256(__file__)},
        parquet_sha256={p.name:file_sha256(p) for p in folder.glob('*.parquet')})
    path.write_text(json.dumps(report,indent=2)+'\n')
    return dataset
