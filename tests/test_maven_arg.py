import json

import pytest
from datasets import Dataset, DatasetDict

from tasksource.tasks import _maven_arg_records, maven_arg__role


def source():
    return dict(input='Alice opened the door.',output=dict(json_structures={
        'Opening':[dict(trigger=dict(text='opened',start=6,end=12),arguments=[
            dict(role='Agent',text='Alice',start=0,end=5),
            dict(role='Theme',text='the door',start=13,end=21)])],
        'Closing':None}))


def test_argument_spans_and_native_roles_without_output_leakage():
    rows=list(_maven_arg_records(source()))
    assert rows[0]['sentence1']=='[ARGUMENT]Alice[/ARGUMENT] [TRIGGER]opened[/TRIGGER] the door.'
    assert rows[1]['sentence1']=='Alice [TRIGGER]opened[/TRIGGER] [ARGUMENT]the door[/ARGUMENT].'
    assert [r['labels'] for r in rows]==['Agent','Theme']
    assert all('Agent' not in r['sentence1']+r['sentence2'] for r in rows)
    assert len({json.loads(r['metadata'])['split_group_id'] for r in rows})==1
    assert len({json.loads(r['metadata'])['id'] for r in rows})==2


def test_ambiguous_duplicate_annotation_is_not_arbitrarily_assigned():
    row=source();args=row['output']['json_structures']['Opening'][0]['arguments']
    args.append(dict(args[0],role='Patient'))
    assert [r['labels'] for r in _maven_arg_records(row)]==['Theme']


def test_bad_offsets_fail_explicitly():
    row=source();row['output']['json_structures']['Opening'][0]['trigger']['start']=5
    with pytest.raises(ValueError,match='does not match'):
        list(_maven_arg_records(row))


def test_native_partitions_and_jev_preserve_argument_question():
    from tasksource.jev.recast import recast_jev
    dataset=DatasetDict(train=Dataset.from_list([source()]),validation=Dataset.from_list([source()]))
    processed=maven_arg__role(dataset)
    assert set(processed)=={'train','validation'}
    assert processed['train'].features['labels'].names==['Agent','Theme']
    # Use the public recast entry point, including the task's question.
    result=recast_jev(processed,task='maven-arg/role',question=maven_arg__role.question)
    row=result['train'][0]
    assert '[ARGUMENT]Alice[/ARGUMENT]' in str(row)
    assert 'text_B' not in row['state']
    assert row['instructions'].endswith('Event type: Opening')
    assert '{sentence2}' not in row['instructions']
