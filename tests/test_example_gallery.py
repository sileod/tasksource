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
    gallery.write(tmp_path, 'pinned', {'known':[row]}, 1)
    page=(tmp_path/'index.html').read_text()
    assert '<script>alert(1)' not in page and '&lt;script&gt;alert(1)' in page
    assert '1. second — 0.25' in page and '2. first — 0.75' in page
    data=json.loads((tmp_path/'examples.json').read_text())
    assert data['revision']=='pinned' and data['tasks'][0]['examples']==[row]
    assert data['tasks'][1]['examples']==[] and 'No training example' in page
    with pytest.raises(ValueError, match='Misaligned'):
        gallery.format_example(dict(row, target=[1.0]))


def test_boolean_and_token_examples_have_readable_gold():
    boolean = dict(id='b', state='Text', question='True?', kind='noul', options=[], target=[0.0])
    assert gallery.format_example(boolean)[-1] == ('Gold target', 'Probability of true: 0')
    tokens = dict(question='Tag the words', canonical=dict(tokens=['Lung', 'disease'], labels=['B-Disease', 'I-Disease']))
    assert gallery.format_example(tokens)[-1] == ('Tokens / gold tags', 'Lung → B-Disease\ndisease → I-Disease')
