import json

from datasets import Dataset,DatasetDict
from tasksource import access,load_task
from tasksource.preprocess import MultipleChoice
from scripts.repackage_dataset.tabmwp import tabmwp_rows


def test_native_choice_setting_preserves_options_and_default_stays_four():
    raw=DatasetDict(train=Dataset.from_list([
        dict(prompt='Question 1',options=[str(i) for i in range(10)],gold=9),
        dict(prompt='Question 2',options=[str(i) for i in range(5)],gold=4)]))
    def task(**kw):return MultipleChoice('prompt',labels='gold',choices_list='options',complete_splits=False,**kw)
    native=task(preserve_choices=True)(raw)
    assert len([k for k in native['train'].features if k.startswith('choice')])==10
    for row in native['train']:
        assert row[f"choice{row['labels']}"]==('9' if row['inputs']=='Question 1' else '4')
    assert native['train'][1]['choice9'] is None
    old=task()(raw)
    assert len([k for k in old['train'].features if k.startswith('choice')])==4
    assert old['train'][0][f"choice{old['train'][0]['labels']}"]=='9'


def test_cbt_native_gold_and_ten_choices_survive_jev(monkeypatch):
    choices=[f'name{i}' for i in range(10)]
    good=dict(sentences=['name9 entered the room.'],question='XXXXX entered.',options=choices,answer='name9')
    bad=dict(good,options=['duplicate']*10,answer='duplicate')
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:DatasetDict(train=Dataset.from_list([good,bad])))
    for task in ('cbt/ne','cbt/cn'):
        rows=load_task(task,recast='jev')['train'];assert len(rows)==1
        row=rows[0];assert len(row['criteria'])==10 and row['criteria'][row['label']]=='name9'
        assert 'Which candidate fills XXXXX' in row['instructions']


def test_recam_reserves_validation_and_unlabeled_test_articles(monkeypatch):
    from tasksource.tasks import _text_hash,_recam_rows
    from tasksource.metadata import recam_test_overlap
    monkeypatch.setattr(recam_test_overlap,'RECAM_TEST_OVERLAP',frozenset({_text_hash('Official test article')}))
    def row(article):return dict(article=article,question='An @placeholder sentence.')
    ds=DatasetDict(train=Dataset.from_list([row('Shared article'),row('Official test article'),row('Only train')]),
                   validation=Dataset.from_list([row('Shared article')]))
    result=_recam_rows(ds)
    assert result['train']['article']==['Only train'] and len(result['validation'])==1


def test_recam_five_choices_and_gold(monkeypatch):
    row=dict(article='A team won the championship.',question='The team celebrated its @placeholder .',label=4,
             **{f'option_{i}':o for i,o in enumerate(['failure','defeat','loss','setback','victory'])})
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:DatasetDict(train=Dataset.from_list([row])))
    r=load_task('recam',recast='jev')['train'][0]
    assert len(r['criteria'])==5 and r['criteria'][r['label']]=='victory'


def test_knowref_native_neither_and_error_filtering(monkeypatch):
    def row(label):return dict(original_sentence=label,swapped_sentence='Amy called Beth because [she] was late.',
        candidate0='Amy',candidate1='Beth',correct_candidate=label,annotation_strength='4')
    raw=DatasetDict(train=Dataset.from_list([row(v) for v in ['candidate_0','candidate_1','neither','error']]+[dict(row('candidate_0'),annotation_strength='3')]))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:raw)
    rows=load_task('knowref60k',recast='jev')['train'];assert len(rows)==3
    for r in rows:
        meta=json.loads(r['metadata'])
        assert r['criteria'][r['label']]=={'candidate_0':'Amy','candidate_1':'Beth','neither':'neither candidate'}[meta['native_label']]
        assert meta['annotation_strength']=='4' and 'error' not in r['criteria']


def test_tabmwp_reserves_free_answer_holdout_tables_and_keeps_solution_private():
    def row(table,choices):return dict(table=table,question='Which day?',choices=choices,answer='Friday',
                                      solution='SECRET_SOLUTION',split='train',grade=2)
    splits=dict(test={'t':row('Held-out table',[])},validation={},train={
        'bad':row('Held-out table',['Friday','Sunday']), 'ok':row('Train table',['Friday','Sunday'])})
    rows,excluded,_=tabmwp_rows(splits)
    assert len(rows['train'])==1 and rows['train'][0]['labels']==0
    assert 'SECRET_SOLUTION' not in rows['train'][0]['inputs']
    assert json.loads(rows['train'][0]['metadata'])['solution']=='SECRET_SOLUTION'
    assert {r['reason'] for r in excluded}=={'not_multiple_choice','heldout_table'}
