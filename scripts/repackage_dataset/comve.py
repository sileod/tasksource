"""Native human ComVE statement pairs, with the sensible statement as gold."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from datasets import Features, Sequence, Value
from .relevance import text_key, write_mirror

SOURCE = 'https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation'
REVISION = 'c55b1a9e6cb66997675f85073e17a9522cf38c2c'
FILES = {
    'train': ('Training  Data', 'subtaskA_data_all.csv', 'subtaskA_answers_all.csv'),
    'validation': ('Dev Data', 'subtaskA_dev_data.csv', 'subtaskA_gold_answers.csv'),
    'test': ('Test Data', 'subtaskA_test_data.csv', 'subtaskA_gold_answers.csv'),
}


def join_answers(data, answers):
    rows=list(csv.DictReader(io.StringIO(data)))
    gold_rows=list(csv.reader(io.StringIO(answers)));gold=dict(gold_rows)
    if len(gold)!=len(gold_rows) or len({r['id'] for r in rows})!=len(rows) or set(gold)!={r['id'] for r in rows}:
        raise ValueError('ComVE requires unique, matching data and answer IDs')
    if any(v not in ('0','1') for v in gold.values()):raise ValueError('Invalid ComVE gold index')
    return [dict(id=r['id'],choices=[r['sent0'],r['sent1']],labels=1-int(gold[r['id']])) for r in rows]


def prepare_pairs(splits):
    """Reserve native holdouts; exclude duplicate pairs and contradictory judgments."""
    judgments=defaultdict(set)
    def keys(row):return [text_key(c) for c in row['choices']]
    for rows in splits.values():
        for row in rows:judgments[tuple(sorted(keys(row)))].add(keys(row)[row['labels']])
    output={};seen=set();excluded=[];counts={}
    for split in ('test','validation','train'):
        output[split]=[];reasons=Counter()
        for row in splits[split]:
            pair=tuple(sorted(keys(row)));group=hashlib.sha256(json.dumps(pair).encode()).hexdigest()
            reason=('invalid_choices' if any(not c.strip() for c in row['choices']) or len(set(pair))!=2 else
                    'conflicting_gold' if len(judgments[pair])>1 else 'duplicate_pair' if pair in seen else None)
            if reason:
                reasons[reason]+=1;excluded.append(dict(split=split,id=row['id'],reason=reason));continue
            output[split].append(dict(inputs='Which statement is more sensible in ordinary circumstances?',
                choices_list=row['choices'],labels=row['labels'],metadata=json.dumps(dict(
                    id=f"{split}:{row['id']}",split_group_id=group,native_split=split,
                    native_nonsensical_index=1-row['labels'],annotation_origin='human'),sort_keys=True)))
            seen.add(pair)
        counts[split]=dict(reasons)
    return output,excluded,counts


def comve():
    """Join official ComVE subtask-A CSVs, invert nonsense labels, reserve native holdouts.

    Original distribution: https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation
    Original README declares CC BY-SA 4.0. No generated choices or negatives.
    """
    splits={};hashes={}
    for split,(folder,data,answers) in FILES.items():
        contents=[]
        for name in (data,answers):
            path=f'ALL data/{folder}/{name}'
            url=SOURCE.replace('github.com','raw.githubusercontent.com')+f'/{REVISION}/'+quote(path,safe='/')
            with urlopen(url,timeout=60) as response:raw=response.read()
            hashes[path]=hashlib.sha256(raw).hexdigest();contents.append(raw.decode('utf-8-sig'))
        splits[split]=join_answers(*contents)
    rows,excluded,counts=prepare_pairs(splits)
    report=dict(source=SOURCE,revision=REVISION,source_sha256=hashes,license='cc-by-sa-4.0',
        annotation_origin='human',native_counts={s:len(r) for s,r in splits.items()},excluded_counts=counts,
        conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        configuration=dict(gold='invert native nonsensical index',split_policy='native; test > validation > train'))
    features=Features(dict(inputs=Value('string'),choices_list=Sequence(Value('string')),
                           labels=Value('int64'),metadata=Value('string')))
    dataset=write_mirror('comve',rows,'int64',report,features)
    Path('build/comve-release/excluded-questions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in excluded))
    return dataset
