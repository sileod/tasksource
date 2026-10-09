import json

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
import pytest

from tasksource import load_task
from tasksource.vision_tasks import grouped_qa_rows, _SUPERCLEVR_SHAPE
from scripts.repackage_dataset.superclevr import canonical_qa, SHAPES


def question(terminal='query_shape', answer='utility bike', output='utility'):
    return dict(program=[{'type': terminal, '_output': output}], answer=answer,
                question='What shape is it?', question_index=1, question_family_index=2)


def test_native_ontology_and_shape_alias():
    assert list(_SUPERCLEVR_SHAPE) == list(SHAPES)
    qa = canonical_qa(question())
    assert qa['view'] == 'shape' and qa['labels'] == 'utility bike'
    assert json.loads(qa['metadata'])['terminal_type'] == 'query_shape'
    assert 'utility' not in qa['inputs']


@pytest.mark.parametrize('terminal,answer,view,label', [
    ('exist', False, 'yesno', 'no'), ('equal_integer', True, 'yesno', 'yes'),
    ('count', 10, 'count', '10'), ('query_color', 'red', 'color', 'red'),
    ('query_size', 'large', 'size', 'large'), ('query_material', 'metal', 'material', 'metal'),
])
def test_program_typed_views(terminal, answer, view, label):
    qa = canonical_qa(question(terminal, answer, answer))
    assert (qa['view'], qa['labels']) == (view, label)


@pytest.mark.parametrize('row', [question(output='road'), question('count', 11, 11),
                                question('exist', 'yes', True), question('query_color', 'red', 'blue')])
def test_bad_program_answers_rejected(row):
    with pytest.raises(ValueError):
        canonical_qa(row)


@pytest.mark.parametrize('view', ['yesno', 'count', 'color', 'shape', 'size', 'material'])
def test_grouped_classification_and_jev(monkeypatch, view):
    import io
    from PIL import Image as PILImage
    buffer = io.BytesIO()
    PILImage.new('RGB', (2, 2), 'red').save(buffer, format='PNG')
    encoded = {'bytes': buffer.getvalue(), 'path': None}
    qas = [canonical_qa(question()), canonical_qa(question('exist', False, False))]
    terminal, answer = {'yesno': ('exist', False), 'count': ('count', 3),
        'color': ('query_color', 'red'), 'shape': ('query_shape', 'utility bike'),
        'size': ('query_size', 'small'), 'material': ('query_material', 'rubber')}[view]
    qas.append(canonical_qa(question(terminal, answer, 'utility' if view == 'shape' else answer)))
    features = Features({'images': Sequence(Image(decode=False)), 'image_group_id': Value('string'),
        'qa': [{'inputs': Value('string'), 'labels': Value('string'), 'view': Value('string'), 'metadata': Value('string')}]})
    source = DatasetDict(train=Dataset.from_list([{'images': [encoded], 'image_group_id': 'native:1',
                                                 'qa': qas}], features=features))
    flattened = grouped_qa_rows(source, view)
    assert all(json.loads(r['metadata'])['image_group_id'] == 'native:1' for r in flattened['train'])
    monkeypatch.setattr('tasksource.access.load_dataset', lambda *a, **kw: DatasetDict(source))
    result = load_task('superclevr/' + view, vision=True, recast='jev')
    assert set(result) == {'train'}
    assert result['train'][0]['images'][0].getpixel((0, 0)) == (255, 0, 0)
