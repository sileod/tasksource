from datasets import Dataset,DatasetDict
from tasksource.tasks import _ems_rejection,_ems_rows


def row(answer='f',**extra):
    return dict(question='A native six-option question?',answer=answer,
        choices=[f'{chr(97+i)}. Answer {i}' for i in range(6)],**extra)


def test_native_single_answer_and_permutation_safe_options():
    assert _ems_rejection(row()) is None
    assert _ems_rejection(row("['a', 'b']"))=='multiple_or_invalid_answer'
    changed=row();changed['choices'][-1]='f. all of the above'
    assert _ems_rejection(changed)=='positional_option_reference'
    changed=row();changed['choices'][1]='c. wrong prefix'
    assert _ems_rejection(changed)=='invalid_option_prefix'


def test_collection_aliases_are_excluded_and_holdouts_win():
    source=row();ds=Dataset.from_list([source]);other=row();other['question']='Different training question?'
    data=DatasetDict(train=Dataset.from_list([source,other]),validation=ds,test=ds,full_data=ds,train_woNA=ds)
    result=_ems_rows(data)
    assert set(result)=={'train','validation','test'}
    assert len(result['train'])==1 and len(result['validation'])==0 and len(result['test'])==1
    assert len(result['train'][0]['choices'])==6
