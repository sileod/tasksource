"""MATRES native main-axis labels joined to the author's TempEval3 documents."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from datasets import Features, Value
from tasksource.tasks import _decision_metadata, _marked_spans
from .linguistic import _download
from .relevance import write_mirror

MATRES_REVISION = 'ccac302b546401a1e55f2bd6097913083b5cd8d6'
TEMPEVAL_REVISION = 'd6518a5a79d940d749c95ee5b72830714b132238'


def timeml_document(raw):
    root=ET.fromstring(raw);parts=[];spans={};position=0
    def append(text):
        nonlocal position
        parts.append(text or '');position+=len(text or '')
    def visit(node):
        start=position;append(node.text)
        for child in node:visit(child);append(child.tail)
        if node.tag=='EVENT':
            content=''.join(parts)[start:position];left=len(content)-len(content.lstrip())
            spans[node.attrib['eid']]=(start+left,position-(len(content)-len(content.rstrip())))
    visit(root.find('TEXT'))
    instances={n.attrib['eiid']:spans[n.attrib['eventID']] for n in root.iter('MAKEINSTANCE')}
    return ''.join(parts),instances


def matres():
    """MATRES verb-event start-point relations, with full original document context.

    Original annotations: https://github.com/qiangning/MATRES.
    Original author-hosted texts: https://github.com/qiangning/StructTempRel-EMNLP17/tree/master/data/TempEval3.
    Join by event INSTANCE ID (eiid), not event ID (eid). TimeBank and AQUAINT
    remain training; Platinum remains test. No random validation split or
    unannotated negative pairs. Original news text rights remain unspecified;
    the availability of annotations is not a grant over the source articles.
    """
    labels={};hashes={}
    for name in ('timebank','aquaint','platinum'):
        url=f'https://raw.githubusercontent.com/qiangning/MATRES/{MATRES_REVISION}/{name}.txt'
        raw=_download(name+'.txt',url);hashes[url]=hashlib.sha256(raw).hexdigest()
        labels[name]=list(csv.reader(io.StringIO(raw.decode()),delimiter='\t'))
    jobs=[]
    for name,rows in labels.items():
        directory={'timebank':'Training/TBAQ-cleaned/TimeBank','aquaint':'Training/TBAQ-cleaned/AQUAINT',
                   'platinum':'Evaluation/te3-platinum'}[name]
        for doc in sorted({r[0] for r in rows}):
            url=f'https://raw.githubusercontent.com/qiangning/StructTempRel-EMNLP17/{TEMPEVAL_REVISION}/data/TempEval3/{directory}/{doc}.tml'
            jobs.append((doc,url))
    def read(job):
        doc,url=job;raw=_download(doc+'.tml',url)
        return doc,timeml_document(raw),url,hashlib.sha256(raw).hexdigest()
    docs={}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for doc,parsed,url,digest in pool.map(read,jobs):docs[doc]=parsed;hashes[url]=digest
    splits=defaultdict(list);excluded=Counter();seen={}
    for name,rows in labels.items():
        split='test' if name=='platinum' else 'train'
        for doc,verb1,verb2,id1,id2,label in rows:
            text,instances=docs[doc]
            if 'ei'+id1 not in instances or 'ei'+id2 not in instances:
                excluded['missing_event_instance']+=1;continue
            a=instances['ei'+id1];b=instances['ei'+id2]
            if text[slice(*a)]!=verb1 or text[slice(*b)]!=verb2:
                raise ValueError(f'MATRES event instance/mention mismatch: {doc}/{id1}/{id2}')
            if label not in {'BEFORE','AFTER','EQUAL','VAGUE'}:raise ValueError('Unknown MATRES label')
            identity=f'{doc}:{id1}:{id2}'
            if identity in seen:
                if seen[identity]!=label:raise ValueError('Conflicting MATRES annotations')
                excluded['duplicate_pair']+=1;continue
            seen[identity]=label
            if max(a[0],b[0])<min(a[1],b[1]):excluded['overlapping_event_spans']+=1;continue
            splits[split].append(dict(sentence1=_marked_spans(text,[('A',*a),('B',*b)]),labels=label,
                metadata=_decision_metadata(identity,doc,event_instance_ids=['ei'+id1,'ei'+id2],event_spans=[a,b],source_partition=name)))
    groups={s:{json.loads(r['metadata'])['split_group_id'] for r in rows} for s,rows in splits.items()}
    assert not groups['train'] & groups['test']
    report=dict(source='https://github.com/qiangning/MATRES',revision=MATRES_REVISION,
        text_source='https://github.com/qiangning/StructTempRel-EMNLP17',text_revision=TEMPEVAL_REVISION,
        source_hashes=hashes,license='unspecified',excluded=dict(excluded),document_counts={s:len(g) for s,g in groups.items()},
        split_policy='native TimeBank+AQUAINT train, Platinum test; no fabricated validation',
        conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    features=Features({k:Value('string') for k in ('sentence1','labels','metadata')})
    return write_mirror('matres',splits,'string',report,features)
