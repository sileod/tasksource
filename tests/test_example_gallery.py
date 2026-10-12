import json

import pandas as pd
import pytest
from scripts import write_example_gallery as gallery


def test_gallery_preserves_order_soft_gold_and_escapes_source_html(tmp_path, monkeypatch):
    monkeypatch.setattr(gallery, 'list_tasks', lambda: pd.DataFrame([
        dict(id='known', task_type='Classification', dataset_name='org/source'),
        dict(id='missing', task_type='SequenceToSequence', dataset_name='org/other')]))
    row=dict(source='known', id='source:1', state='<script>alert(1)</script>\n```',
             question='Which?', kind='choice', options=['second', 'first'], target=[.25, .75], split='train')
    rows = [dict(row, id=f'source:{i}') for i in range(3)]
    (tmp_path / 'tasks').mkdir()
    orphan = tmp_path / 'tasks/task-retired.md'
    orphan.write_text('Retired task')
    gallery.write(tmp_path, 'pinned', {'known': rows}, 3)
    assert not orphan.exists()
    page=(tmp_path/'index.html').read_text()
    assert '<script>alert(1)' not in page and '&lt;script&gt;alert(1)' in page
    assert '1. second — 0.25' in page and '2. first — 0.75' in page
    data=json.loads((tmp_path/'examples.json').read_text())
    assert data['revision']=='pinned' and data['tasks'][0]['examples']==rows
    assert data['tasks'][1]['examples']==[] and 'No training example' in page
    assert page.count('<section class="primary-example">') == 1
    assert page.count('<details class="extra-example">') == 2
    assert '<summary>Example 1</summary>' not in page
    with pytest.raises(ValueError, match='Misaligned'):
        gallery.format_example(dict(row, target=[1.0]))


def test_boolean_and_token_examples_have_readable_gold():
    boolean = dict(id='b', state='Text', question='True?', kind='noul', options=[], target=[0.0])
    assert gallery.format_example(boolean)[-1] == ('Gold target', 'Probability of true: 0')
    tokens = dict(question='Tag the words', canonical=dict(tokens=['Lung', 'disease'], labels=['B-Disease', 'I-Disease']))
    assert gallery.format_example(tokens)[-1] == ('Tokens / gold tags', 'Lung → B-Disease\ndisease → I-Disease')


def test_long_examples_keep_ends_and_complete_text_available():
    value = 'BEGIN' + '<long>' * 400 + 'END'
    rendered = gallery.html_text(value)
    assert 'BEGIN' in rendered and 'END' in rendered and '[middle omitted]' in rendered
    assert 'Full text (' in rendered and '<long>' not in rendered
    assert gallery.html_text('short') == '<pre>short</pre>'


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
