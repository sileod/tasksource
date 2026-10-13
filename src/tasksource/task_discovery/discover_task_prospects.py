"""Cheap discovery screening; outputs a handoff, never implements or parks tasks.

Input: canonical HF/PWC/other retrieval JSONL snapshots. Output: full matching
report, sampled candidates, source evidence, per-model reviews and prospects.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlencode

from .identifiers import normalized,source_id
from .link import cached_json
from .match import catalog_hash,catalog_index,catalog_references,match,previous_decisions,queued_sources
from .jev_review import MODEL,run


def is_train_split(name):
    return bool(re.match(r'^(train|training)($|[_/-])',name.casefold()))


def training_evidence(row,cache,offline=False):
    evidence=[];errors=[];splits=[];complete=True
    for repo in row.get('huggingface',[]):
        repo=repo.split('/datasets/',1)[-1].strip('/')
        url='https://datasets-server.huggingface.co/splits?'+urlencode(dict(dataset=repo))
        try:
            response=cached_json(url,cache,offline);data=response['data']
            evidence.append(dict(url=url,sha256=response['sha256'],metadata=data))
            source_splits=[s for s in data.get('splits',[]) if not row.get('config_name') or s['config']==row['config_name']]
            splits.extend(source_splits);complete &= not data.get('pending') and not data.get('failed')
            train=next((s for s in source_splits if is_train_split(s['split'])),None)
            if train:
                url='https://datasets-server.huggingface.co/rows?'+urlencode(dict(
                    dataset=repo,config=train['config'],split=train['split'],offset=0,length=1))
                preview=cached_json(url,cache,offline)
                evidence.append(dict(url=url,sha256=preview['sha256'],metadata=preview['data']))
        except (OSError,ValueError) as error:errors.append(dict(url=url,error=str(error)))
    has_train=any(is_train_split(s['split']) for s in splits)
    status='present' if has_train else 'not_listed' if splits and complete and not errors else 'unknown'
    split_names=sorted({s['split'] for s in splits})
    policy='train_only' if has_train and complete and not errors and all(is_train_split(s) for s in split_names) else 'multiple_splits' if has_train and len(split_names)>1 else 'unknown'
    return dict(row,train_status=status,source_splits=split_names,split_policy=policy,evidence=row.get('evidence',[])+evidence,training_check_errors=errors)


def current_answers(row):
    return [r.get('data',{}).get('answers',{}) for r in row.get('reviews',[]) if r.get('reviewer')=='jev_review'
            and not r.get('failed') and r.get('state_sha256')==row.get('review_state_sha256')
            and r.get('questions_sha256')==row.get('review_questions_sha256')]


def observed_resource(row):
    return any(d.get('url','').startswith('https://datasets-server.huggingface.co/rows?')
               and d.get('sha256') and d.get('metadata',{}).get('rows') for d in row.get('evidence',[]))


def prospect_status(row):
    if row.get('review_status')=='failed' or not row.get('review_evidence_count'):return 'needs_evidence'
    answers=current_answers(row)
    if not answers:return 'needs_evidence'
    # Keep each model's decision: the average alone cannot erase disagreement.
    questions=('train','decision_fit') if observed_resource(row) else ('train','resources','decision_fit')
    if row.get('review_languages'):questions+=('dataset_language',)
    for question in questions:
        if all(a.get(question,{}).get('probabilities',{}).get('no',0)>=0.8 for a in answers):return 'not_suitable'
    if row.get('train_status')=='not_listed' or (row.get('huggingface') and row.get('train_status')!='present'):
        return 'needs_evidence'
    if row.get('training_check_errors'):return 'needs_evidence'
    if all(a.get(q,{}).get('probabilities',{}).get('yes',0)>=0.8 for a in answers
           for q in questions):return 'promising'
    return 'needs_evidence'


def origin_matches(row,allowed,allow_verified_synthetic=False):
    """Keep allowed origins, optionally admitting independently verified synthetic labels."""
    if not allowed:return True
    answers=current_answers(row)
    for answer in answers:
        origin=answer.get('annotation_origin',{}).get('probabilities',{})
        checks=answer.get('annotation_checks',{}).get('probabilities',{})
        covered=answer.get('verification_scope',{}).get('probabilities',{}).get('all',0)>=0.8
        verified=sum(checks.get(k,0) for k in ('independent_review','agreement','executable','multiple'))>=0.8
        if sum(origin.get(k,0) for k in set(allowed))<0.8 and not (allow_verified_synthetic
                and origin.get('model',0)+origin.get('mixed',0)>=0.8 and covered and verified):return False
    return bool(answers)


def save_prospects(rows,output,origins=(),allow_verified_synthetic=False):
    prospects=[dict(row,resource_observed=bool(observed_resource(row)),prospect_status=prospect_status(row)) for row in rows]
    promising=[r for r in prospects if r['prospect_status']=='promising']
    handoff=[r for r in promising if origin_matches(r,origins,allow_verified_synthetic)]
    write_rows(output/'prospects.jsonl',prospects)
    write_rows(output/'handoff.jsonl',handoff)
    return dict(status_counts={s:sum(r['prospect_status']==s for r in prospects)
                               for s in ('promising','needs_evidence','not_suitable')},
                annotation_origins=origins,allow_verified_synthetic=allow_verified_synthetic,handoff_count=len(handoff),annotation_filtered=len(promising)-len(handoff))


def language_priority(row, requested):
    """Prefer declared matches; keep unknown language, exclude explicit mismatches."""
    aliases={'english':'en'}
    languages=row.get('languages') or []
    if isinstance(languages,str):languages=[languages]
    known={aliases.get(v.casefold(),v.casefold()) for v in languages}
    wanted={aliases.get(v.casefold(),v.casefold()) for v in requested}
    return 0 if not wanted or known & wanted else 1 if not known else 2


def unseen_candidates(rows,previous):
    """Exclude already screened identities, without merging shared papers/GitHub repos."""
    names={normalized(r['name']) for r in previous}
    repos={source_id(v,'huggingface') for r in previous for v in r.get('huggingface',[])}
    return [r for r in rows if normalized(r['name']) not in names
            and not repos.intersection(source_id(v,'huggingface') for v in r.get('huggingface',[]))]


def write_rows(path,rows):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in rows));temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=100);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--cache',type=Path,default=Path('build/discovery/review-cache'))
    parser.add_argument('--language',action='append',default=[],help='Prefer this dataset language; retain unknown, exclude known mismatches')
    parser.add_argument('--exclude',action='append',type=Path,default=[],help='Prior screened JSONL; repeat to avoid spending on the same candidates')
    parser.add_argument('--annotation-origin',action='append',default=[],choices=('human','procedural','model','mixed','unknown'),help='Allowed label origins in handoff; repeat for OR, all reviewers must support them')
    parser.add_argument('--allow-verified-synthetic',action='store_true',help='With an origin filter, also admit model/mixed labels with independent checks covering all retained labels')
    parser.add_argument('--from-reviews',action='store_true',help='Filter saved reviews without network or model calls')
    parser.add_argument('--include-known',action='store_true',help='Bypass catalog/history/queue exclusion for calibration')
    parser.add_argument('--model',action='append');parser.add_argument('--offline-evidence',action='store_true')
    args=parser.parse_args()
    if args.allow_verified_synthetic and not args.annotation_origin:args.annotation_origin=['human','procedural']
    if args.limit<1:parser.error('--limit must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for path in args.inputs for line in path.read_text().splitlines() if line.strip()]
    if args.from_reviews:
        manifest=save_prospects(rows,args.output,args.annotation_origin,args.allow_verified_synthetic)
        manifest.update(input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs},from_reviews=True)
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(manifest));return
    names,repos=catalog_index();references=catalog_references();history=previous_decisions();queue=queued_sources()
    matched=list(match(rows,names,repos,history,queue,references))
    write_rows(args.output/'matched.jsonl',matched)
    candidates=[r for r in matched if args.include_known or r['status']=='candidate']
    previous=[json.loads(line) for p in args.exclude for line in p.read_text().splitlines() if line.strip()]
    unseen=unseen_candidates(candidates,previous)
    eligible=[r for r in unseen if language_priority(r,args.language)<2]
    # Reproducible source-stratified sample, not just the highest-download sources.
    groups={}
    for row in eligible:
        source=tuple(sorted(s['source'] for s in row.get('sources',[])))
        groups.setdefault(source,[]).append(row)
    for group in groups.values():group.sort(key=lambda r:(language_priority(r,args.language),hashlib.sha256(f"{args.seed}:{normalized(r['name'])}".encode()).digest()))
    selected=[]
    while groups and len(selected)<args.limit:
        for source in sorted(list(groups)):
            if len(selected)==args.limit:break
            selected.append(groups[source].pop(0))
            if not groups[source]:del groups[source]
    write_rows(args.output/'selected.jsonl',selected)
    print(f'Selected {len(selected)} candidates; checking training metadata',flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        checked=list(pool.map(lambda r:training_evidence(r,args.cache,args.offline_evidence),selected))
    write_rows(args.output/'checked.jsonl',checked)
    print('Training metadata saved; collecting evidence and reviewing',flush=True)
    manifest=run(checked,args.output/'review',args.cache,args.model or [MODEL],args.offline_evidence,languages=args.language) if checked else {}
    reviewed=[json.loads(l) for l in (args.output/'review/reviews.jsonl').read_text().splitlines()] if checked else []
    manifest.update(save_prospects(reviewed,args.output,args.annotation_origin,args.allow_verified_synthetic))
    manifest.update(input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs},
        catalog_sha256=catalog_hash(names,repos),
        reference_sha256=hashlib.sha256(json.dumps({k:sorted(v) for k,v in references.items()},sort_keys=True).encode()).hexdigest(),
        decisions_sha256=hashlib.sha256(json.dumps(history,sort_keys=True).encode()).hexdigest(),
        queue_sha256=hashlib.sha256(json.dumps(queue,sort_keys=True).encode()).hexdigest(),
        discovery_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        exclusion_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.exclude},
        previously_screened=len(candidates)-len(unseen),
        retrieved=len(rows),unique=len(matched),novel=sum(r['status']=='candidate' for r in matched),
        selected=len(selected),seed=args.seed,languages=args.language,language_eligible=len(eligible),include_known=args.include_known)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':main()
