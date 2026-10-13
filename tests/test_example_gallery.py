import json

import pandas as pd
import pytest
from scripts import write_example_gallery as gallery


def test_catalog_is_one_parseable_example_per_task_with_soft_gold(tmp_path, monkeypatch):
    import re, yaml
    monkeypatch.setattr(gallery, 'list_tasks', lambda: pd.DataFrame([
        dict(id='known', task_type='Classification', dataset_name='org/source'),
        dict(id='missing', task_type='Classification', dataset_name='org/other')]))
    row = dict(source='known', id='source:1', state='<b>x</b>\n```', question='Which?', kind='choice',
               options=['second', 'first'], target=[.25, .75], split='train')
    out = tmp_path / 'examples.md'
    gallery.write(out, 'pinned', {'known': [row, dict(row, id='source:2')]})
    page = out.read_text()
    blocks = [yaml.safe_load(b) for b in re.findall(r"````yaml\n(.*?)\n````", page, re.S)]
    assert blocks == [dict(task='known', state='<b>x</b>\n```', question='Which?', options=['second', 'first'],
                           answer={'second': .25, 'first': .75})]
    assert '`missing`' in page and 'pinned' in page
    with pytest.raises(ValueError, match='Misaligned'):
        gallery.format_example(dict(row, target=[1.0]))


def test_boolean_and_token_examples_have_readable_gold():
    boolean = dict(id='b', state='Text', question='True?', kind='noul', options=[], target=[0.0])
    assert gallery.format_example(boolean)[-1] == ('Gold target', 'Probability of true: 0')
    tokens = dict(question='Tag the words', canonical=dict(tokens=['Lung', 'disease'], labels=['B-Disease', 'I-Disease']))
    assert gallery.format_example(tokens)[-1] == ('Tokens / gold tags', 'Lung → B-Disease\ndisease → I-Disease')


def test_long_texts_are_cropped_in_the_middle():
    value = 'BEGIN' + 'x' * 5000 + 'END'
    cropped = gallery.crop(value)
    assert cropped.startswith('BEGIN') and cropped.endswith('END') and 'characters omitted' in cropped
    assert len(cropped) < 1300 and gallery.crop('short') == 'short'


def test_canonical_fallback_keeps_full_vocabulary_and_renders_paired_state(monkeypatch):
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
    data = DatasetDict(train=Dataset.from_dict({'sentence1': ['Premise'], 'sentence2': ['Claim'], 'labels': [0]},
        features=Features({'sentence1': Value('string'), 'sentence2': Value('string'),
                           'labels': ClassLabel(names=['entailment', 'neutral', 'contradiction'])})))
    monkeypatch.setattr('tasksource.load_task', lambda *a, **kw: data)
    monkeypatch.setattr('tasksource.task_provenance', lambda task: {})
    row = gallery.source_examples('demo', 1)[0]
    assert row['criteria'] == ['entailment', 'neutral', 'contradiction']
    fields = dict(gallery.format_example(row))
    assert fields['State'] == 'text_A: Premise\ntext_B: Claim'
    assert 'entail' in fields['Question']
    assert fields['Criteria (label vocabulary)'].splitlines() == row['criteria']
    assert fields['Gold labels'] == 'entailment'


def test_token_fallback_shows_question_and_all_tags_including_unobserved(monkeypatch):
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
    data = DatasetDict(train=Dataset.from_dict({'tokens': [['Fever']], 'labels': [[1]]},
        features=Features({'tokens': Sequence(Value('string')),
                           'labels': Sequence(ClassLabel(names=['O', 'B-Disease', 'I-Disease']))})))
    data.question = 'Tag disease mentions using BIO labels.'
    monkeypatch.setattr('tasksource.load_task', lambda *a, **kw: data)
    monkeypatch.setattr('tasksource.task_provenance', lambda task: {})
    row = gallery.source_examples('demo', 1)[0]
    fields = dict(gallery.format_example(row))
    assert fields['Question'] == data.question
    assert fields['Criteria (label vocabulary)'] == 'O\nB-Disease\nI-Disease'
    assert fields['Tokens / gold tags'] == 'Fever → B-Disease'
