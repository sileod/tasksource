"""Resumable, bounded-memory expansion of the reviewed procedural generators."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import contextlib
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import pyarrow.parquet as pq
from datasets import IterableDataset, IterableDatasetDict

from .common import file_sha256, seed_for
from ..vision import _materialize

TRON_TASKS = ('aquarium_grid','angle_bisector','bar_chart_aggregate','before_after','overlapping_shape_count')


def chunk(job):
    name,index,directory,settings = job
    from scripts.upload_repackaged import BUILDERS,load_prepared
    builder = BUILDERS[name][1]
    directory.mkdir(parents=True,exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(directory)
        release = Path('build')/(name+'-release')
        if (release/'provenance.json').exists():
            load_prepared(release,'tasksource/'+name)
        else:
            with Path('run.log').open('a') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                builder(**settings)
            shutil.rmtree(release/'cache',ignore_errors=True)  # Checkpoint Parquet is sufficient for resumption.
        return str(directory/release)
    finally:
        os.chdir(previous)


def collect(directory, groups, fingerprints, cap, excluded):
    """Index small QA records; image bytes remain in checkpoint Parquet files."""
    for split in ('validation','train'):
        path = Path(directory)/(split+'.parquet')
        if not path.exists():continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                digest = row['image_group_id']
                if split=='train' and digest in groups['validation']:
                    excluded['image_in_validation']+=len(row['qa']);continue
                for qa in row['qa']:
                    choices,label = qa['choices_list'],qa['labels']
                    metadata = json.loads(qa['metadata'])
                    if len(choices)<2 or len(set(choices))!=len(choices) or not 0<=label<len(choices):
                        raise ValueError('Invalid checkpoint choices')
                    if metadata['verifier_scores']!=[int(i==label) for i in range(len(choices))]:
                        raise ValueError('Invalid checkpoint gold alignment')
                    key = hashlib.sha256(json.dumps([digest,qa['inputs'],choices[label]]).encode()).hexdigest()
                    if key in fingerprints[split]:excluded['duplicate_question']+=1;continue
                    if split=='train' and len(fingerprints[split])>=cap:
                        excluded['train_cap']+=1;continue
                    fingerprints[split].add(key)
                    groups[split].setdefault(digest,[]).append(qa)


def examples(directories, split, groups):
    seen = set()
    for directory in directories:
        path = Path(directory)/(split+'.parquet')
        if not path.exists():continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                identity = row['image_group_id']
                if identity not in groups or identity in seen:continue
                seen.add(identity)
                yield dict(images=row['images'],image_group_id=identity,qa=groups[identity])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name',choices=('tron-mcq','mentis-oculi-paper-fold','sphinx-mcq','visgym-mcq'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train',type=int,default=100000)
    parser.add_argument('--eval',type=int,default=100)
    parser.add_argument('--chunk-size',type=int,default=250)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--push',action='store_true')
    args = parser.parse_args()
    if min(args.train,args.eval,args.chunk_size,args.workers)<1:parser.error('Counts must be positive')
    from scripts.upload_repackaged import BUILDERS,load_prepared,push_card
    builder = BUILDERS[args.name][1]
    root = args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    files = [Path(__file__),Path(__file__).with_name('common.py'),Path(inspect.getsourcefile(builder))]
    config = dict(source=args.name,train=args.train,eval=args.eval,chunk_size=args.chunk_size,seed=args.seed,
                  task_ids=list(TRON_TASKS) if args.name=='tron-mcq' else [],scripts_sha256={p.name:file_sha256(p) for p in files})
    manifest = root/'configuration.json'
    if manifest.exists() and json.loads(manifest.read_text())!=config:
        raise ValueError('Resume configuration changed; use a new output directory')
    manifest.write_text(json.dumps(config,indent=2)+'\n')
    def job(index):
        settings = dict(max_rows=args.chunk_size,max_rows_eval=args.eval if index==0 else 0,seed=seed_for(args.seed,index))
        if args.name=='tron-mcq':settings['task_ids']=TRON_TASKS
        return args.name,index,root/'chunks'/f'{index:05d}',settings
    started = time.monotonic();groups = {'validation':{},'train':{}};fingerprints = {s:set() for s in groups}
    directories = [chunk(job(0))];excluded = Counter()
    collect(directories[0],groups,fingerprints,args.train,excluded)
    maximum = (2*args.train+args.chunk_size-1)//args.chunk_size
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for start in range(1,maximum,args.workers):
            if len(fingerprints['train'])>=args.train:break
            wave = pool.map(chunk,[job(i) for i in range(start,min(start+args.workers,maximum))])
            for directory in wave:
                directories.append(directory);collect(directory,groups,fingerprints,args.train,excluded)
                elapsed = time.monotonic()-started;count = len(fingerprints['train'])
                status = dict(train=count,target=args.train,completed_chunks=len(directories),elapsed_seconds=round(elapsed),
                              eta_seconds=round((args.train-count)*elapsed/count) if count else None)
                (root/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(status,flush=True)
    if len(fingerprints['train'])<args.train:
        raise ValueError('Candidate budget exhausted before reaching distinct verified training target; checkpoints retained')
    from datasets import Dataset
    features = Dataset.from_parquet(str(Path(directories[0])/'train.parquet')).features
    # Datasets interprets list-valued gen_kwargs as shards; keep one iterator's dedup state.
    dataset = IterableDatasetDict({s:IterableDataset.from_generator(examples,gen_kwargs=dict(directories=tuple(directories),split=s,groups=g),features=features) for s,g in groups.items() if g})
    report = json.loads((Path(directories[0])/'provenance.json').read_text())
    previous = Path.cwd()
    try:
        os.chdir(root)
        dataset = _materialize(dataset,args.name,report['source'],report['revision'],[],main,report['license'],report['license_url'])
        release = root/'build'/(args.name+'-release');provenance = release/'provenance.json'
        checkpoints = [json.loads((Path(d)/'provenance.json').read_text()) for d in directories]
        (release/'generation-checkpoints.json').write_text(json.dumps(checkpoints,indent=2)+'\n')
        combined = {**report,**json.loads(provenance.read_text()),'configuration':config,
                    'checkpoint_provenance':[dict(path=str(Path(d).relative_to(root)/'provenance.json'),sha256=file_sha256(Path(d)/'provenance.json')) for d in directories],
                    'scale_exclusions':dict(excluded),
                    'generation_checkpoints_sha256':file_sha256(release/'generation-checkpoints.json'),
                    'image_overlap_excluded':sum(r['image_overlap_excluded'] for r in checkpoints)+excluded['image_in_validation'],
                    'exclusion_counts':dict(sum((Counter(r.get('exclusion_counts',{})) for r in checkpoints),Counter())),
                    'parquet_sha256':{p.name:file_sha256(p) for p in release.glob('*.parquet')}}
        combined.pop('generation_exclusions',None)  # Full per-chunk reasons are in generation-checkpoints.json.
        for split,info in combined['splits'].items():
            if info['rows']!=len(groups[split]) or info['questions']!=len(fingerprints[split]):
                raise ValueError('Materialized counts differ from the deduplicated checkpoint index')
        if (root/'resume-correction.json').exists():
            combined['resume_correction']=json.loads((root/'resume-correction.json').read_text())
        notices=Path(directories[0])/'asset-notices.json'
        if notices.exists():shutil.copy2(notices,release/'asset-notices.json')
        provenance.write_text(json.dumps(combined,indent=2)+'\n')
        if args.push:
            dataset = load_prepared(release,'tasksource/'+args.name)
            dataset.push_to_hub('tasksource/'+args.name)
            push_card('tasksource/'+args.name,builder,release)
            from huggingface_hub import HfApi
            revision = HfApi().dataset_info('tasksource/'+args.name).sha
            (root/'published.json').write_text(json.dumps(dict(repo='tasksource/'+args.name,revision=revision,splits=combined['splits']),indent=2)+'\n')
    finally:
        os.chdir(previous)


if __name__=='__main__':main()
