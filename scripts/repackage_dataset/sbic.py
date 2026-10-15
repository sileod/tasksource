"""Conservative SBIC judgments with the original crowd annotations retained."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from datasets import Features, Value, load_dataset
from huggingface_hub import hf_hub_download
from .relevance import text_key, write_mirror

SOURCE = 'allenai/social_bias_frames'
REVISION = '453e07c6887c7420e7c62e3bbc1c03ceb4ab0f68'  # official Parquet conversion
DISAGREEMENT_SOURCE = 'RuyuanWan/SBIC_Disagreement'
DISAGREEMENT_REVISION = '01bcc51d26814e610d91eef4ecf39df687babc63'
LABELS = {
    'offensiveness': ('offensiveYN', {'0.0':'not offensive','0.5':'uncertain','1.0':'offensive'}),
    'intent': ('intentYN', {'0.0':'definitely not intended to offend','0.33':'probably not intended to offend',
                          '0.66':'probably intended to offend','1.0':'definitely intended to offend'}),
}
ANNOTATION_FIELDS = ('WorkerId','HITId','offensiveYN','intentYN','whoTarget','sexYN',
                     'targetCategory','targetMinority','targetStereotype','dataSource')


def aggregate_sbic(dataset, min_votes=3, min_agreement=.75, reserved_splits=None):
    """Count a worker once per post/field; conflicting worker ratings abstain."""
    if min_votes < 3 or not .5 < min_agreement <= 1:
        raise ValueError('Require at least three workers and agreement above 50%')
    reserved_splits=reserved_splits or {}
    output={};seen=set();excluded=[];report={}
    for split in ('test','validation','train'):
        groups=defaultdict(list)
        for index,row in enumerate(dataset[split]):
            groups[text_key(row['post'])].append(dict(source_index=f'{split}:{index}',**row))
        output[split]=[];reasons=Counter();accepted={view:Counter() for view in LABELS}
        for group,rows in groups.items():
            if group in seen:
                reasons['cross_split_duplicate']+=1
                excluded.append(dict(split=split,group=group,reason='cross_split_duplicate',
                                     source_indices=[r['source_index'] for r in rows]))
                continue
            conflicts=[s for s,posts in reserved_splits.items() if s!=split and group in posts]
            if conflicts:
                reasons['existing_task_split_conflict']+=1
                excluded.append(dict(split=split,group=group,reason='existing_task_split_conflict',
                                     conflicting_splits=conflicts,source_indices=[r['source_index'] for r in rows]))
                continue
            labels={};aggregation={}
            for view,(field,ontology) in LABELS.items():
                workers=defaultdict(set)
                for r in rows:
                    if r['WorkerId'] and r[field]:
                        if r[field] not in ontology:raise ValueError(f'Unknown SBIC {field}: {r[field]}')
                        workers[r['WorkerId']].add(r[field])
                counts=Counter(next(iter(v)) for v in workers.values() if len(v)==1)
                total=sum(counts.values());gold,n=counts.most_common(1)[0] if counts else (None,0)
                status='accepted' if total>=min_votes and n/total>=min_agreement else (
                    'insufficient_votes' if total<min_votes else 'disagreement')
                labels[view]=ontology[gold] if status=='accepted' else None
                aggregation[view]=dict(counts=dict(counts),valid_workers=total,
                    conflicting_workers=sum(len(v)>1 for v in workers.values()),
                    agreement=n/total if total else 0,status=status)
                reasons[view+'/'+status]+=1
                if labels[view] is not None:accepted[view][labels[view]]+=1
            if not any(v is not None for v in labels.values()):
                excluded.append(dict(split=split,group=group,reason='no_reliable_label',aggregation=aggregation,
                                     source_indices=[r['source_index'] for r in rows]))
                continue
            metadata=dict(id=group,split_group_id=group,source_split=split,source_revision=REVISION,
                annotation_origin='human',aggregation=aggregation,
                annotations=[{k:r[k] for k in ('source_index',*ANNOTATION_FIELDS)} for r in rows])
            variants=sorted({r['post'] for r in rows})
            if len(variants)>1:metadata['post_variants']=variants
            output[split].append(dict(sentence1=rows[0]['post'],**labels,
                                      metadata=json.dumps(metadata,sort_keys=True)))
        # Reserve even rejected holdout posts: they must not re-enter via train.
        seen.update(groups)
        report[split]=dict(native_rows=len(dataset[split]),
                           native_rows_sha256=hashlib.sha256(json.dumps(list(dataset[split]),sort_keys=True).encode()).hexdigest(),
                           post_groups=len(groups),
                           retained_posts=len(output[split]),decisions=dict(reasons),
                           label_counts={k:dict(v) for k,v in accepted.items()})
    return output,report,excluded


def sbic(min_votes=3, min_agreement=.75):
    """SBIC reliable crowd judgments, one post per row with two nullable labels.

    sentence1 is the original post. offensiveness and intent are nullable
    categorical labels; filter out nulls for the desired task. metadata is a
    JSON string with the original annotation rows and aggregation statistics.

    At least three distinct workers and 75% agreement on a native category are
    required per view (normally unanimity with three workers). Repeat stereotype
    rows never add votes; conflicting ratings by the same worker abstain for that
    field. Uncertain/probably ratings remain their original categories, not
    probabilities. A missing group-target answer is not a negative label.
    Group-target ratings and rationales are retained only in metadata: a coarse
    review found a unanimous but questionable negative. whoTarget=1 means
    identity-group targeting; the legacy Hub card reverses this mapping.
    Native splits are retained; normalized duplicate posts favor test, validation,
    then train, even when the holdout lacks a reliable label. Posts conflicting
    with the existing SBIC_Disagreement task splits are excluded, not moved. Metadata retains
    source row IDs, hashed worker IDs, original ratings, target annotations and
    per-view vote counts/agreement/exclusion reasons. Annotation demographics are
    not model inputs. Original annotations: CC BY 4.0; posts retain source terms.
    Rebuild: python scripts/upload_repackaged.py sbic
    """
    data=load_dataset(SOURCE,revision=REVISION)
    reserved={};hashes={}
    for split in ('train','validation','test'):
        path=Path(hf_hub_download(DISAGREEMENT_SOURCE,split+'.csv',repo_type='dataset',revision=DISAGREEMENT_REVISION))
        hashes[split+'.csv']=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as stream:reserved[split]={text_key(r['text']) for r in csv.DictReader(stream)}
    rows,report,excluded=aggregate_sbic(data,min_votes,min_agreement,reserved)
    features=Features({k:Value('string') for k in ('sentence1',*LABELS,'metadata')})
    result=write_mirror('sbic',rows,None,dict(source='https://huggingface.co/datasets/'+SOURCE,
        revision=REVISION,annotation_origin='human',license='cc-by-4.0',
        config=dict(min_votes=min_votes,min_agreement=min_agreement,version=1),audit=report,
        existing_task_holdouts=dict(source=DISAGREEMENT_SOURCE,revision=DISAGREEMENT_REVISION,source_hashes=hashes),
        group_target_mapping={'0.0':'individual or non-identity group','1.0':'identity group'},
        deferred_view='group targeting: questionable unanimous negative in the coarse source audit',
        annotation_instructions='https://maartensap.com/social-bias-frames/annotationTask.html',
        conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),features=features)
    Path('build/sbic-release/excluded-questions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in excluded))
    return result
