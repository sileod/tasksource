"""Pinned entity sentiment and query relevance mirrors; no generated labels or negatives."""
import ast
from collections import Counter
import csv
import hashlib
import heapq
import io
import json
from pathlib import Path
import re
import urllib.request

from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import snapshot_download
import pyarrow.parquet as pq

from .text import _split_of

SENTFIN_REVISION = 'eba43610fe3cb57a3e5773cd6c037da1f8992ad5'
WANDS_REVISION = '16cff4fbe4b46ee0b382e7901ee62461f88a2cb1'
SEARCH_REVISION = '781d35d1bf87253b3dcd0fadcb82bfbee9c244f1'


def text_key(text):
    return hashlib.sha256(' '.join(text.casefold().split()).encode()).hexdigest()


def parse_sentfin(value):
    """Repair only the released unescaped-apostrophe dictionary representation."""
    try:
        result = ast.literal_eval(value)
        repaired = False
    except (ValueError, SyntaxError):
        pattern = r"'(.*?)'\s*:\s*'(positive|neutral|negative)'(?=\s*[,}])"
        pairs = re.findall(pattern, value)
        remainder = re.sub(pattern, '', value).strip()
        if not pairs or re.sub(r'[{},\s]', '', remainder) or len(dict(pairs)) != len(pairs):
            raise ValueError('Unrecognized SEntFiN annotation representation')
        result, repaired = dict(pairs), True
    if not isinstance(result, dict) or not result or any(
        not isinstance(k, str) or not k.strip() or v not in ('negative', 'neutral', 'positive')
        for k, v in result.items()):
        raise ValueError('Invalid SEntFiN entity sentiment')
    return result, repaired


def write_mirror(name, splits, label_type, report, features=None):
    directory = Path('build') / (name + '-release')
    directory.mkdir(parents=True, exist_ok=True)
    features = features or Features(dict(sentence1=Value('string'), sentence2=Value('string'),
                                         labels=Value(label_type), metadata=Value('string')))
    data = DatasetDict({split:Dataset.from_list(rows,features=features) if rows else
        Dataset.from_dict({key:[] for key in features},features=features) for split,rows in splits.items()})
    report.setdefault('conversion_sha256',hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report.update(splits={s:len(d) for s,d in data.items()})
    if label_type in ('string','int64'):
        report['label_counts']={s:dict(Counter(str(r['labels']) for r in rows)) for s,rows in splits.items()}
    for split, rows in data.items():
        rows.to_parquet(str(directory / (split + '.parquet')))
    (directory/'excluded-questions.jsonl').write_text('')
    (directory/'provenance.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return data


def sentfin():
    """SEntFiN entity sentiment, one native headline/entity judgment per row.

    Pinned original CSV: https://github.com/pyRis/SEntFiN (repository MIT license).
    Only malformed dictionary quoting is repaired; the original annotation string
    and source row ID remain in metadata. Identical normalized headlines stay in
    one deterministic 80/10/10 internal partition. These are derived holdouts, not
    an official benchmark split. Duplicate pairs are deduplicated; contradictory
    judgments for the same headline/entity are excluded and reported.
    """
    url=f'https://raw.githubusercontent.com/pyRis/SEntFiN/{SENTFIN_REVISION}/SEntFiN.csv'
    raw=urllib.request.urlopen(url).read()
    records, repairs, conflicts, duplicates = {}, [], set(), 0
    for row in csv.DictReader(io.StringIO(raw.decode())):
        labels, repaired = parse_sentfin(row['Decisions'])
        if repaired:
            repairs.append(dict(id=row['S No.'], original=row['Decisions'], parsed=labels))
        group=text_key(row['Title'])
        for entity,label in labels.items():
            key=(group,entity.casefold())
            if key in records:
                if records[key]['labels'] != label:
                    conflicts.add(key)
                else:
                    duplicates += 1
                continue
            meta=dict(id=f"{row['S No.']}:{entity}", headline_id=row['S No.'], entity=entity,
                split_group_id=group, source_url=url, source_revision=SENTFIN_REVISION,
                source_annotation=row['Decisions'], quoting_repaired=repaired)
            records[key]=dict(sentence1=row['Title'],sentence2=entity,labels=label,
                              metadata=json.dumps(meta,ensure_ascii=False,sort_keys=True))
    splits={s:[] for s in ('train','validation','test')}
    for (group,entity),row in records.items():
        if (group,entity) not in conflicts:
            splits[_split_of(group)].append(row)
    return write_mirror('sentfin',splits,'string',dict(source_url=url,revision=SENTFIN_REVISION,
        source_sha256=hashlib.sha256(raw).hexdigest(),repairs=repairs,duplicate_pairs=duplicates,
        excluded_conflicting_pairs=[list(k) for k in sorted(conflicts)],
        split_policy='normalized-headline-hash-80/10/10',license='mit',
        license_url=f'https://github.com/pyRis/SEntFiN/blob/{SENTFIN_REVISION}/LICENSE'))


def wands():
    """WANDS native three-grade product relevance with query-disjoint internal holdouts.

    Original Wayfair data: https://github.com/wayfair/WANDS (MIT).
    Uses the pinned joined napsternxg/wands Hub copy; its random row splits overlap
    on queries and are replaced by deterministic 80/10/10 normalized-query groups.
    All native judgments are retained. Inputs contain query and product content;
    popularity/rating fields and relevance labels never enter model inputs.
    """
    root=Path(snapshot_download('napsternxg/wands',repo_type='dataset',revision=WANDS_REVISION,
                               allow_patterns=['data/*.parquet']))
    splits={s:[] for s in ('train','validation','test')}
    counts=Counter()
    for path in sorted(root.glob('data/*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            for row in batch.to_pylist():
                if row['label'] not in (0,1,2):
                    raise ValueError('Unknown WANDS relevance grade')
                group=text_key(row['query'])
                fields=('product_name','product_class','category hierarchy','product_description','product_features')
                product='\n'.join(f'{key}: {row[key]}' for key in fields if row.get(key))
                meta=dict(id=str(row['id']),query_id=row['query_id'],product_id=row['product_id'],
                    split_group_id=group,original_mirror_split=path.name.split('-')[0],
                    source_dataset='napsternxg/wands',source_revision=WANDS_REVISION)
                splits[_split_of(group)].append(dict(sentence1=row['query'],sentence2=product,
                    labels=row['label'],metadata=json.dumps(meta,sort_keys=True)))
                counts[row['label']] += 1
    return write_mirror('wands',splits,'int64',dict(source='napsternxg/wands',revision=WANDS_REVISION,
        original_source='https://github.com/wayfair/WANDS',label_counts=dict(counts),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('data/*.parquet')},
        split_policy='normalized-query-hash-80/10/10',license='mit',
        license_url='https://github.com/wayfair/WANDS/blob/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5/LICENSE'))


def search_record(row, candidate):
    score=candidate['score']
    if not isinstance(score,(int,float)) or not 0 <= score <= 14:
        raise ValueError(f'Unexpected SciRepEval click-derived score: {score}')
    document='\n'.join(v for v in (candidate.get('title'),candidate.get('abstract')) if v)
    if not document.strip() or not row['query'].strip():
        return None
    group=text_key(row['query'])
    meta=dict(id=f"{row['doc_id']}:{candidate['doc_id']}",query_id=row['doc_id'],
        candidate_id=candidate['doc_id'],split_group_id=group,source_revision=SEARCH_REVISION,
        source_score=score,supervision='click-derived score, not a human relevance rating',
        score_anchors=list(range(0,15,2)),score_interpolation='linear; expected score equals native value')
    return dict(sentence1=row['query'],sentence2=document,labels=float(score),metadata=json.dumps(meta,sort_keys=True))


def scirepeval_search(max_rows=5000, max_rows_eval=500, seed=42):
    """SciRepEval Search native train/validation pairs with raw click-derived numeric scores.

    No evaluation/test benchmark data is consumed. All training/validation shards
    are scanned; deterministic hash sampling covers eligible pairs from the entire
    source, not an early prefix. Validation queries seen in training are excluded.
    Scores remain unchanged. Jev uses linear interpolation over eight anchors
    0,2,...,14 to preserve the numeric expectation within its 10-level limit.
    Dataset/text licensing is unspecified: the code's Apache license is not used
    as a grant for paper abstracts. Evidence and exclusions remain in provenance.
    """
    root=Path(snapshot_download('allenai/scirepeval',repo_type='dataset',revision=SEARCH_REVISION,
        allow_patterns=['search/train-*.parquet','search/validation-*.parquet'],max_workers=4))
    splits, counts, train_groups = {}, Counter(), set()
    scores=Counter()
    for split,limit in [('train',max_rows),('validation',max_rows_eval)]:
        pool=[]
        for path in sorted(root.glob(f'search/{split}-*.parquet')):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                for source in batch.to_pylist():
                    group=text_key(source['query'])
                    if split=='validation' and group in train_groups:
                        counts['validation/excluded/query_overlap'] += len(source['candidates'])
                        continue
                    if split=='train':
                        train_groups.add(group)
                    for candidate in source['candidates']:
                        record=search_record(source,candidate)
                        if record is None:
                            counts[split+'/excluded/missing_text'] += 1
                            continue
                        counts[split+'/eligible'] += 1
                        scores[candidate['score']] += 1
                        identity=f"{source['doc_id']}:{candidate['doc_id']}"
                        rank=-int(hashlib.sha256(f'{seed}:{identity}'.encode()).hexdigest(),16)
                        item=(rank,identity,counts[split+'/eligible'],record)
                        if len(pool)<limit:
                            heapq.heappush(pool,item)
                        elif item[:2]>pool[0][:2]:
                            heapq.heapreplace(pool,item)
        # Duplicate native query/candidate pairs must not become duplicated training examples.
        selected={identity:record for _,identity,_,record in sorted(pool)}
        splits[split]=list(selected.values())
    return write_mirror('scirepeval-search',splits,'float64',dict(source='allenai/scirepeval',
        revision=SEARCH_REVISION,config='search',counts=dict(counts),source_score_counts=dict(scores),
        selection=dict(max_rows=max_rows,max_rows_eval=max_rows_eval,seed=seed,method='full-source hash sample'),
        split_policy='native train/validation; validation queries in train excluded',license='unspecified'))


ANTIQUE_HASHES = {
    "antique-collection.txt": "68b6688f5f2668c93f0e8e43384f66def768c4da46da4e9f7e2629c1c47a0c36",
    "antique-train-queries.txt": "ced8c5d03883aba18524bd679a2e2f529284d58b224f4b9749ea6e1e1bfcac4d",
    "antique-test-queries.txt": "72e52d256534a0d5ca6ef8b6a745231317b28025e842a3f1fc6a90260213b4f1",
    "antique-train.qrel": "9a07e782666628d3317f0039d682075d800959dc42e97f494c5d7f1d8faef821",
    "antique-test.qrel": "d5585c6812e69ce86c81c406e3688d2cbbe55cf0a079924e894afa4fd62bdbcd",
    "readme.txt": "88ed48a6c84ebb46513cf97f80d24a053d801a696c7a6e8f9a754d7911e6d551"
}


def antique():
    """All native ANTIQUE judged pairs, retaining relevance grades 1–4.

    Join the original query, answer-collection and qrel files without generated
    negatives or filtering grades. Preserve native query-disjoint train/test.
    Q0/E0/U0 judgment provenance stays in metadata. The original readme permits
    research use; Yahoo answer rights are not replaced by a software license.
    Original distribution: https://ciir.cs.umass.edu/downloads/Antique/
    """
    base='https://ciir.cs.umass.edu/downloads/Antique/'
    directory=Path('build/retrieval-expansion');directory.mkdir(parents=True,exist_ok=True)
    files={}
    for name,expected in ANTIQUE_HASHES.items():
        path=directory/name
        if not path.exists():path.write_bytes(urllib.request.urlopen(base+name,timeout=120).read())
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError(f'ANTIQUE source changed: {name}')
        files[name]=raw.decode('utf-8')
    documents=dict(line.split('\t',1) for line in files['antique-collection.txt'].splitlines())
    output={};groups={}
    for split in ('train','test'):
        queries=dict(line.split('\t',1) for line in files[f'antique-{split}-queries.txt'].splitlines())
        output[split]=[];groups[split]=set(queries);seen=set()
        for line in files[f'antique-{split}.qrel'].splitlines():
            query,judgment,document,grade=line.split();grade=int(grade)
            if grade not in (1,2,3,4) or (query,document) in seen:raise ValueError('Invalid or duplicate ANTIQUE judgment')
            seen.add((query,document))
            meta=dict(id=query+':'+document,split_group_id=query,query_id=query,document_id=document,judgment_type=judgment)
            output[split].append(dict(sentence1=f"Question: {queries[query]}\nCandidate answer: {documents[document]}",
                sentence2='',labels=grade,metadata=json.dumps(meta,sort_keys=True)))
    assert not groups['train']&groups['test'],'ANTIQUE query overlap'
    assert len(output['train'])==27422 and len(output['test'])==6589
    return write_mirror('antique',output,'int64',dict(source_url=base,source_hashes=ANTIQUE_HASHES,
        split_policy='native query-disjoint train/test; no fabricated validation',excluded={},
        judgment_types={s:dict(Counter(json.loads(r['metadata'])['judgment_type'] for r in rows)) for s,rows in output.items()},
        conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),license='research-only; original ANTIQUE distribution'))
