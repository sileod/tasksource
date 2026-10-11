"""Original argumentation, stance and discourse data, preserving native supervision."""
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from datasets import Features, Sequence, Value

from tasksource.metadata.discourse import DISCO_RELATIONS
from .relevance import text_key, write_mirror

PINS = {
    'emilyallaway/zero-shot-stance':'e7c4775182b184730f350995f8260579c9e066fe',
    'FakeNewsChallenge/fnc-1':'29d473af2d15278f0464d5e41e4cbe7eb58231f2',
    'CogComp/perspectrum':'b42c27b85f20f2a35c0a0895698fa242ccf160ab',
    'UKPLab/acl2016-convincing-arguments':'ca9d24e41b8805ff2e1d3585d552b49bb64a65f7',
    'merelscholman/DiscoGeM':'95ea9a45681bfdbe7ecbd66fcaf03f42d5141468',
}


def _source(repo):
    report=dict(source='https://github.com/'+repo,revision=PINS[repo],source_hashes={},
                conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def read(path):
        url=f'https://raw.githubusercontent.com/{repo}/{PINS[repo]}/{quote(path)}'
        cache=Path('build/source-archives')/repo/PINS[repo]/path
        if not cache.exists():
            cache.parent.mkdir(parents=True,exist_ok=True)
            cache.write_bytes(urlopen(url,timeout=60).read())
        raw=cache.read_bytes();report['source_hashes'][path]=hashlib.sha256(raw).hexdigest()
        return raw.decode('utf-8-sig')
    return read,report


def _row(a,b,label,identity,group,**metadata):
    return dict(sentence1=a,sentence2=b,labels=label,
                metadata=json.dumps(dict(id=str(identity),split_group_id=group,**metadata),sort_keys=True))


def _holdouts(splits,report):
    """Keep native holdouts when identical source groups occur in several partitions."""
    seen=set();excluded=Counter()
    for split in ('test','validation','train'):
        if split not in splits:continue
        groups={json.loads(r['metadata'])['split_group_id'] for r in splits[split]}
        keep=[r for r in splits[split] if json.loads(r['metadata'])['split_group_id'] not in seen]
        excluded[split]=len(splits[split])-len(keep);splits[split]=keep;seen.update(groups)
    report['excluded_group_overlap']=dict(excluded)
    return splits


def vast():
    """VAST native human stance annotations, using original comments and topics.

    Excludes type_idx=4 synthetic neutral pairs. Keeps native train/dev/test,
    complete original comments, annotation IDs and article IDs. Original data:
    https://github.com/emilyallaway/zero-shot-stance/tree/master/data/VAST.
    Dataset license is unspecified; no code license is applied to the comments.
    """
    read,report=_source('emilyallaway/zero-shot-stance');splits={};excluded={}
    for split,native in [('train','train'),('validation','dev'),('test','test')]:
        rows=list(csv.DictReader(io.StringIO(read(f'data/VAST/vast_{native}.csv'))))
        excluded[split]=sum(r['type_idx']=='4' for r in rows)
        splits[split]=[_row(r['post'],r['new_topic'],r['label'],r['new_id'],text_key(r['post']),
            article_id=r['arc_id'],annotation_type=r['type_idx'],seen_topic=r['seen?']) for r in rows if r['type_idx']!='4']
        if any(r['labels'] not in ('0','1','2') for r in splits[split]):raise ValueError('Unknown VAST label')
    report.update(excluded_synthetic_neutral=excluded,split_policy='native comment-disjoint partitions',license='unspecified')
    return write_mirror('vast',_holdouts(splits,report),'string',report)


def fnc1():
    """FNC-1 native headline/body stance, joined from the original labeled CSVs.

    Preserves complete article bodies and official training/competition-test
    partitions. Does not invent validation data. Identical body copies remain
    in the native holdout. Original distribution: https://github.com/FakeNewsChallenge/fnc-1.
    The original distribution does not specify a dataset license.
    """
    read,report=_source('FakeNewsChallenge/fnc-1');splits={}
    for split,prefix in [('train','train'),('test','competition_test')]:
        bodies={r['Body ID']:r['articleBody'] for r in csv.DictReader(io.StringIO(read(prefix+'_bodies.csv')))}
        splits[split]=[]
        for i,r in enumerate(csv.DictReader(io.StringIO(read(prefix+'_stances.csv')))):
            if r['Stance'] not in ('agree','disagree','discuss','unrelated'):raise ValueError('Unknown FNC stance')
            body=bodies[r['Body ID']]
            splits[split].append(_row(r['Headline'],body,r['Stance'],f'{split}:{i}',text_key(body),body_id=r['Body ID']))
    report.update(split_policy='native train/competition-test, body-disjoint',license='unspecified')
    return write_mirror('fnc1',_holdouts(splits,report),'string',report)


def perspectrum():
    """PERSPECTRUM native claim/perspective stance with original claim partitions.

    Joins the official perspective pool; uses the first original perspective in
    each annotated cluster, without expanding paraphrases or inventing negatives.
    Original distribution: https://github.com/CogComp/perspectrum/tree/master/data/dataset.
    Dataset terms: CC BY-SA 4.0, recorded in the original data/dataset/license.txt.
    """
    read,report=_source('CogComp/perspectrum');base='data/dataset/'
    pool={r['pId']:r['text'] for r in json.loads(read(base+'perspective_pool_v1.0.json'))}
    native=json.loads(read(base+'dataset_split_v1.0.json'));splits={s:[] for s in ('train','validation','test')}
    for claim in json.loads(read(base+'perspectrum_with_answers_v1.0.json')):
        split=native[str(claim['cId'])];split='validation' if split=='dev' else split
        for annotation in claim['perspectives']:
            label=annotation['stance_label_3']
            if label not in ('SUPPORT','UNDERMINE'):raise ValueError('Unknown PERSPECTRUM stance')
            pid=annotation['pids'][0]
            splits[split].append(_row(claim['text'],pool[pid],label,f"{claim['cId']}:{pid}",text_key(claim['text']),
                claim_id=claim['cId'],perspective_id=pid,native_cluster=annotation['pids'],votes=annotation['voter_counts']))
    report.update(split_policy='native claim partitions',license='cc-by-sa-4.0')
    return write_mirror('perspectrum',_holdouts(splits,report),'string',report)


def ukpconvarg():
    """UKPConvArg1Strict native two-argument convincingness judgments (CC BY 4.0).

    Original source: https://github.com/UKPLab/acl2016-convincing-arguments.
    No official train/test partition: deterministically hash-ranked topics
    allocate 12 debates to training and two each to validation and test, keeping
    both stance sides and all recurring arguments of a debate together.
    These are internal holdouts, not the paper's cross-validation folds.
    """
    repo='UKPLab/acl2016-convincing-arguments';read,report=_source(repo)
    tree=json.loads(urlopen(f'https://api.github.com/repos/{repo}/git/trees/{PINS[repo]}?recursive=1',timeout=60).read())
    paths=sorted(r['path'] for r in tree['tree'] if r['path'].startswith('data/UKPConvArg1Strict-CSV/') and r['path'].endswith('.csv'))
    topics=sorted({Path(p).stem.split('_',1)[0] for p in paths},key=text_key)
    holdout=max(1,round(len(topics)*.1))
    partitions={topic:'test' if i<holdout else 'validation' if i<2*holdout else 'train' for i,topic in enumerate(topics)}
    splits={s:[] for s in ('train','validation','test')}
    for path in paths:
        topic=Path(path).stem.split('_',1)[0];split=partitions[topic]
        for r in csv.DictReader(io.StringIO(read(path)),delimiter='\t'):
            if r['label'] not in ('a1','a2') or not r['a1'].strip() or not r['a2'].strip() or r['a1']==r['a2']:
                raise ValueError('Invalid strict convincingness pair')
            metadata=json.dumps(dict(id=Path(path).stem+':'+r['#id'],split_group_id=topic,native_pair=r['#id']),sort_keys=True)
            splits[split].append(dict(inputs='Debate topic: '+topic.replace('-',' '),choices_list=[r['a1'],r['a2']],
                                      labels=int(r['label']=='a2'),metadata=metadata))
    report.update(split_policy='derived hash-ranked topics, two validation/two test',topic_partitions=partitions,license='cc-by-4.0')
    features=Features(dict(inputs=Value('string'),choices_list=Sequence(Value('string')),labels=Value('int64'),metadata=Value('string')))
    return write_mirror('ukpconvarg',splits,'int64',report,features)


def discourse_votes(value, observations):
    votes={k:float(v) for k,v in (part.strip().split(':') for part in value.split(';'))}
    counts={k:round(v*observations) for k,v in votes.items()}
    if (set(votes)!=set(DISCO_RELATIONS) or any(v<0 for v in votes.values())
        or any(abs(v*observations-counts[k])>.015 for k,v in votes.items()) or sum(counts.values())!=observations):
        raise ValueError('Invalid native DiscoGeM vote distribution')
    return [counts[k] for k in DISCO_RELATIONS]


def discogem():
    """DiscoGeM 1.0 native crowd vote shares and original train/dev/test labels.

    Recovers integer votes from rounded majority_softlabel and observations;
    validates their sum. No sampled tie-breaking, IRT or model probabilities.
    Keeps context-containing argument spans and actual observation counts. Rows
    marked exclude are omitted. Shared source paragraphs remain in their native
    highest-priority holdout (test, dev, train); this empties the overlapping dev
    partition rather than fabricating one. Original distribution: https://github.com/merelscholman/DiscoGeM.
    Underlying source texts have mixed terms; overall dataset license is unspecified.
    """
    read,report=_source('merelscholman/DiscoGeM');splits={s:[] for s in ('train','validation','test')};excluded=0
    for r in csv.DictReader(io.StringIO(read('DiscoGeM 1.0/DiscoGeM 1.0_items/DiscoGeM1.0.wide.csv'))):
        if r['split']=='exclude':excluded+=1;continue
        split='validation' if r['split']=='dev' else r['split']
        splits[split].append(_row(r['arg1'],r['arg2'],discourse_votes(r['majority_softlabel'],int(r['observations'])),r['itemid'],r['datasource'],
            observations=int(r['observations']),genre=r['genre'],native_votes=r['majority_softlabel']))
    report.update(excluded_native=excluded,split_policy='native paragraph partitions',license='unspecified',vote_order=list(DISCO_RELATIONS))
    splits={s:rows for s,rows in _holdouts(splits,report).items() if rows}
    report['vote_counts']={s:dict(zip(DISCO_RELATIONS,map(sum,zip(*(r['labels'] for r in rows))))) for s,rows in splits.items()}
    features=Features(dict(sentence1=Value('string'),sentence2=Value('string'),labels=Sequence(Value('float64')),metadata=Value('string')))
    return write_mirror('discogem',splits,'float64',report,features)
