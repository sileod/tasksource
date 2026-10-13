"""Resumable, exhaustive English Hub screening in a fixed priority order."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .hf_hub import retrieve
from .identifiers import normalized
from .jev_review import configure_key,QUESTIONS
from .match import catalog_index,catalog_references,match,previous_decisions,queued_sources
from .discover_task_prospects import unseen_candidates,write_rows

STAGES={
    'classification':('text-classification','token-classification'),
    'zero-shot-classification':('zero-shot-classification',),
    'sentence-similarity':('sentence-similarity',),
    'ranking':('text-ranking','text-retrieval'),
}
VISUAL_STAGES={
    'multiple-choice':('multiple-choice',),
    'visual-question-answering':('visual-question-answering',),
    'image-classification':('image-classification',),
}
MODEL='openrouter/~typesafe/jev-latest'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temporary.replace(path)


def control(models,path):
    """Check opposite answers and missing evidence before spending on a batch."""
    from litlm import complete
    from litlm_cli import _record
    configure_key()
    state='Verified facts: training_split_present=true; download_available=false. Annotator identity is not specified.'
    instructions={'train':'Is training_split_present true?', 'resources':'Is download_available true?',
                  'annotators':'Does the evidence establish that medical experts annotated the data?'}
    questions={k:dict(type='choice',instructions=v,criteria=QUESTIONS['train']['criteria']) for k,v in instructions.items()}
    expected={'train':'yes','resources':'no','annotators':'unknown'}
    accepted=[];records=[]
    for model in models:
        try:
            result=complete([state],model=model,decisions=questions,api_key_envs=['JEV_OPENROUTER_API_KEY'],
                            caching=False,num_retries=0,show_progress=False)[0]
            record=_record(result)
            passed=not result.failed and all(result.data['answers'][q]['probabilities'].get(v,0)>=0.8 for q,v in expected.items())
        except Exception as error:
            record=dict(model=model,failed=True,error=dict(type=type(error).__name__,message=str(error)))
            passed=False
        records.append(dict(record,passed=passed))
        if passed:accepted.append(model)
    save_json(path,dict(state=state,questions=questions,expected=expected,results=records))
    return accepted


def report(output,state,stages):
    save_json(output/'status.json',state)
    rows=[];lines=['# Exhaustive discovery queue','',f"Status: {state['status']}. Updated: {state['updated_at']}.",'',
                  '| Stage | Status | Inventory | Selected | Reviewed | Shortlist |', '|---|---|---:|---:|---:|---:|']
    for stage in stages:
        info=state['stages'][stage]
        batches=sorted((output/stage/'batches').glob('*/manifest.json'))
        reviewed=shortlisted=0
        for manifest in batches:
            data=json.loads(manifest.read_text());reviewed+=data.get('selected',0)
            handoff=manifest.parent/'handoff.jsonl'
            if handoff.exists():
                found=read_rows(handoff);rows.extend(found);shortlisted+=len(found)
        lines.append(f"| {stage} | {info['status']} | {info.get('inventory','')} | {info.get('selected','')} | {reviewed} | {shortlisted} |")
    write_rows(output/'handoff.jsonl',rows)
    (output/'report.md').write_text('\n'.join(lines)+'\n\nFull judgments, uncertainties and request failures remain in each batch’s prospects.jsonl. Shortlisted sources still require quality, novelty and leakage checks.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--exclude',type=Path,action='append',default=[])
    parser.add_argument('--model',action='append',help='Default Jev; every configured reviewer must pass the batch control')
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--cache',type=Path,default=Path('build/discovery/cache'))
    parser.add_argument('--stage',action='append',choices=tuple(STAGES)+tuple(VISUAL_STAGES),help='Repeat to set the screening order; defaults to the text queue')
    parser.add_argument('--modality',action='append',default=[],help='HF modality filter, e.g. image')
    args=parser.parse_args()
    if args.batch_size<1:parser.error('--batch-size must be positive')
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    lock=(output/'queue.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:parser.error('This queue is already running')
    models=args.model or [MODEL]
    stages={s:(STAGES|VISUAL_STAGES)[s] for s in args.stage} if args.stage else STAGES
    config=dict(stages=stages,languages=['en'],models=models,batch_size=args.batch_size,
                exclusions={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.exclude},
                questions_sha256=hashlib.sha256(json.dumps(QUESTIONS,sort_keys=True).encode()).hexdigest())
    if args.modality:config['modalities']=args.modality
    config=json.loads(json.dumps(config))
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text())!=config:
        parser.error('Queue settings changed; use a new output directory')
    save_json(output/'config.json',config)
    state=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else dict(
        stages={stage:dict(status='queued') for stage in stages})
    def update(status):
        state.update(status=status,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat())
        report(output,state,stages)
    previous=[row for path in args.exclude for row in read_rows(path)]
    update('running')
    try:
        for stage,tags in stages.items():
            directory=output/stage;directory.mkdir(exist_ok=True)
            info=state['stages'][stage]
            if info['status']=='complete':
                previous.extend(read_rows(directory/'selected.jsonl'));continue
            info['status']='retrieving';update('running')
            if not (directory/'inventory.jsonl').exists() or not (directory/'inventory.manifest.json').exists():
                rows,manifest=retrieve(args.cache,tasks=tags,limit=None,languages=['en'],modalities=args.modality)
                write_rows(directory/'inventory.jsonl',rows);save_json(directory/'inventory.manifest.json',manifest)
            inventory=read_rows(directory/'inventory.jsonl');info['inventory']=len(inventory)
            if not (directory/'selected.jsonl').exists():
                names,repos=catalog_index()
                matched=list(match(inventory,names,repos,previous_decisions(),queued_sources(),catalog_references()))
                write_rows(directory/'matched.jsonl',matched)
                selected=unseen_candidates([r for r in matched if r['status']=='candidate'],previous)
                write_rows(directory/'selected.jsonl',sorted(selected,key=lambda r:normalized(r['name'])))
            selected=read_rows(directory/'selected.jsonl');info.update(selected=len(selected),status='screening')
            update('running')
            for start in range(0,len(selected),args.batch_size):
                batch=directory/'batches'/f'{start//args.batch_size:04d}';batch.mkdir(parents=True,exist_ok=True)
                if (batch/'manifest.json').exists() and (batch/'handoff.jsonl').exists():continue
                healthy=control(models,batch/'control.json')
                if not healthy:
                    info['status']='reviewer_unavailable';update('reviewer_unavailable');return
                write_rows(batch/'input.jsonl',selected[start:start+args.batch_size])
                command=[sys.executable,'-u','-m','tasksource.task_discovery.discover_task_prospects',str(batch/'input.jsonl'),
                         '--limit',str(args.batch_size),'--output',str(batch),'--allow-verified-synthetic']
                if args.modality:command+=['--language','en']
                for model in healthy:command+=['--model',model]
                save_json(batch/'command.json',command)
                with (batch/'run.log').open('a') as log:
                    for attempt in range(2):
                        result=subprocess.run(command,stdout=log,stderr=log,timeout=3600)
                        if result.returncode==0:break
                        if attempt==0:time.sleep(10)
                if result.returncode:
                    info['status']='batch_failed';update('batch_failed');return
                info['last_batch']=start//args.batch_size;update('running')
            info['status']='complete';previous.extend(selected);update('running')
        update('complete')
    except Exception as error:
        state['error']=dict(type=type(error).__name__,message=str(error));update('failed');raise


if __name__=='__main__':main()
