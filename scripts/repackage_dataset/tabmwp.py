"""Native text-table multiple-choice TabMWP; no generated answers or distractors."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from datasets import Features, Sequence, Value
from .relevance import text_key, write_mirror

SOURCE = 'https://github.com/lupantech/PromptPG'
REVISION = '5a1a52214521590b075f545b76a4f5ce666345e3'
FILES = {'train':'train','validation':'dev','test':'test'}


def tabmwp_rows(splits):
    """Reserve entire native held-out tables, including their free-answer questions."""
    output={};seen=set();excluded=[];counts={}
    for split in ('test','validation','train'):
        output[split]=[];reasons=Counter()
        for identifier,row in splits[split].items():
            group=text_key(row['table']);choices=row['choices']
            reason=('not_multiple_choice' if not choices else
                    'heldout_table' if group in seen else
                    'invalid_choices' if len(choices)<2 or any(not c.strip() for c in choices) or
                        len({text_key(c) for c in choices})!=len(choices) or choices.count(row['answer'])!=1 else
                    'missing_table' if not row['table'].strip() else None)
            if reason:
                reasons[reason]+=1;excluded.append(dict(split=split,id=identifier,reason=reason));continue
            title=row.get('table_title')
            inputs='Table'+(f': {title}' if title else ':')+'\n'+row['table']+'\n\nQuestion: '+row['question']
            if row.get('unit'):inputs+='\nAnswer unit: '+row['unit']
            output[split].append(dict(inputs=inputs,choices_list=choices,labels=choices.index(row['answer']),
                metadata=json.dumps(dict(id=identifier,split_group_id=group,native_split=row['split'],
                    grade=row['grade'],solution=row['solution'],annotation_origin='human',source=SOURCE),sort_keys=True)))
        seen.update(text_key(r['table']) for r in splits[split].values())
        counts[split]=dict(reasons)
    return output,excluded,counts


def tabmwp():
    """Native TabMWP MC questions with complete textual tables and original choices.

    Original distribution: https://github.com/lupantech/PromptPG/tree/main/data/tabmwp
    Dataset CC BY-NC-SA 4.0 (distinct from the code's MIT license). Native train/dev/test
    partitions are preserved; tables appearing in holdouts are removed from training.
    Free-answer problems supply no fabricated MC questions. Solutions remain metadata.
    """
    splits={};hashes={}
    for split,name in FILES.items():
        path=f'data/tabmwp/problems_{name}.json'
        with urlopen(SOURCE.replace('github.com','raw.githubusercontent.com')+f'/{REVISION}/{path}',timeout=60) as r:raw=r.read()
        hashes[path]=hashlib.sha256(raw).hexdigest();splits[split]=json.loads(raw)
    rows,excluded,counts=tabmwp_rows(splits)
    report=dict(source=SOURCE,revision=REVISION,source_sha256=hashes,license='cc-by-nc-sa-4.0',
        annotation_origin='human',native_counts={s:len(r) for s,r in splits.items()},excluded_counts=counts,
        conversion_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        configuration=dict(view='native multiple choice, textual table',split_policy='native; test > validation > train by table'))
    features=Features(dict(inputs=Value('string'),choices_list=Sequence(Value('string')),
                          labels=Value('int64'),metadata=Value('string')))
    dataset=write_mirror('tabmwp',rows,'int64',report,features)
    Path('build/tabmwp-release/excluded-questions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in excluded))
    return dataset
