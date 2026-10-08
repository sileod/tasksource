import io
import os
import pytest
from PIL import Image as PILImage
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value, ClassLabel
from tasksource import list_tasks, load_task, task_provenance, hub_datasets
from tasksource.preprocess import VisualClassification, VisualMultipleChoice, Classification
from tasksource.recast import recast_instruct
from tasksource.vision_tasks import figureqa_rows, ai2d_rows


def png(color):
    buffer = io.BytesIO()
    PILImage.new('RGB', (2, 2), color).save(buffer, format='PNG')
    return {'bytes': buffer.getvalue(), 'path': None}


def source(mc=False):
    rows = [{'left': png('red'), 'right': png('blue'), 'text': str(i),
             'answer': 4 if i % 2 else 1, 'options': ['a', 'b', 'c', 'd', 'gold'] if i % 2 else ['a', 'gold']}
            for i in range(6)]
    features = Features({'left': Image(), 'right': Image(), 'text': Value('string'),
                         'answer': Value('int64'), 'options': Sequence(Value('string'))})
    ds = Dataset.from_list(rows, features=features)
    return DatasetDict(train=ds, validation=ds.select([0, 1]))


def template(mc=False, two=False):
    cls = VisualMultipleChoice if mc else VisualClassification
    kwargs = {'choices_list': 'options'} if mc else {'labels': lambda x: 'yes' if x['answer'] == 4 else 'no'}
    return cls(images=lambda x: [x['left'], x['right']] if two else [x['left']],
               inputs='text', **({'labels': 'answer'} if mc else {}), **kwargs)


def test_catalog():
    vision = list_tasks(vision=True)
    assert list(vision.id) == ['nlvr2', 'snli-ve', 'aokvqa', 'scienceqa-img', 'ai2d', 'figureqa',
                             'mind2web/action', 'mind2web/element', 'mind2web/x10',
                             'mind2web/y10', 'mind2web/grid5', 'mind2web/grid7']
    assert not set(vision.id) & set(list_tasks().id)
    assert all(task_provenance(i, vision=True)['revision'] for i in vision.id)
    assert 'pingzhili/nlvr2' in hub_datasets(['nlvr2'], vision=True)


@pytest.mark.parametrize('two', [False, True])
@pytest.mark.parametrize('mc', [False, True])
def test_images_labels_and_splits(mc, two):
    ds = template(mc, two)(source())
    assert set(ds) == {'train', 'validation'}
    assert len(ds['train']) == 6 and len(ds['validation']) == 2
    assert ds['train'].features['images'] == Sequence(Image())
    assert isinstance(ds['train'].features['labels'], ClassLabel)
    raw = ds['train'].cast_column('images', Sequence(Image(decode=False)))[0]['images']
    assert raw == [png('red'), png('blue')] if two else raw == [png('red')]
    if mc:
        assert ds['train'].features['labels'].names == list('ABCDE')
        for row in ds['train']:
            assert row[f"choice{row['labels']}"] == 'gold'
        assert ds['train'][0]['choice4'] is None
        assert ds['train'][1]['choice4'] == 'gold'


@pytest.mark.parametrize('mc', [False, True])
def test_instruct(mc):
    ds = template(mc, True)(source())
    result = recast_instruct(ds, seed=7)
    assert set(result['train'].features) == {'images', 'inputs', 'targets'}
    assert result['train'].features['images'] == Sequence(Image())
    assert result['train'][0]['images'][0].getpixel((0, 0)) == (255, 0, 0)
    assert result['train'].to_dict() == recast_instruct(ds, seed=7)['train'].to_dict()
    if mc:
        for row in result['train']:
            letter = row['targets'][0]
            assert f'{letter}: gold' in row['inputs']
            assert 'None' not in row['inputs']


def test_grouped_sources():
    rows = Dataset.from_list([{'image': png('red'), 'qa': [{'question': 'q1', 'answer': 'Yes.'}, {'question': 'q2', 'answer': 'No.'}]}],
        features=Features({'image': Image(decode=False), 'qa': [{'question': Value('string'), 'answer': Value('string')}]}))
    ds = VisualClassification(pre_process=figureqa_rows)(DatasetDict(train=rows))
    assert set(ds) == {'train'} and len(ds['train']) == 2
    assert ds['train']['inputs'] == ['q1', 'q2']
    ai = Dataset.from_list([{'images': [png('blue')], 'texts': [{'user': 'Question: q\nChoices:\nA. x\nB. y\nAnswer with the letter.', 'assistant': 'Answer: B', 'source': 'AI2D'}]}],
        features=Features({'images': Sequence(Image(decode=False)), 'texts': [{'user': Value('string'), 'assistant': Value('string'), 'source': Value('string')}]}))
    ds = VisualMultipleChoice(choices_list='choices_list', pre_process=ai2d_rows)(DatasetDict(train=ai))
    assert ds['train'][0]['labels'] == 1 and ds['train'][0]['choice1'] == 'y'


def test_text_regression():
    rows = Dataset.from_list([{'text': str(i), 'label': 'yes' if i % 2 else 'no'} for i in range(100)])
    ds = Classification(sentence1='text', labels='label')(DatasetDict(train=rows))
    assert set(ds) == {'train', 'validation', 'test'}
    assert set(recast_instruct(ds)['train'].features) == {'inputs', 'targets'}


@pytest.mark.skipif(not os.getenv('TASKSOURCE_VISION_SMOKE'), reason='opt-in Hub downloads')
@pytest.mark.parametrize('task_id', list(list_tasks(vision=True).id))
def test_real_sources(task_id):
    ds = load_task(task_id, vision=True, streaming=True, max_rows=10, max_rows_eval=10)
    assert len(ds['train']) > 0
    assert ds['train'].features['images'] == Sequence(Image())
    assert isinstance(ds['train'].features['labels'], ClassLabel)


def test_loader_and_metadata(monkeypatch):
    import tasksource.access as access
    from tasksource import task_licenses
    rows = Dataset.from_list([{'image0': png('red'), 'image1': png('blue'), 'sentence': 'q', 'label': 'True', 'identifier': 'train-1-0-0'}],
        features=Features({'image0': Image(), 'image1': Image(), 'sentence': Value('string'), 'label': Value('string'), 'identifier': Value('string')}))
    calls = []
    def loader(name, config, **kwargs):
        calls.append((name, kwargs))
        return DatasetDict(train=rows)
    monkeypatch.setattr(access, 'load_dataset', loader)
    ds = load_task('nlvr2', vision=True, max_rows=10)
    assert ds['train'].features['labels'].names == ['False', 'True']
    assert ds['train'][0]['labels'] == 1
    assert calls[0][1]['revision'] == task_provenance('nlvr2', vision=True)['revision']
    assert set(task_licenses(vision=True).id) == set(list_tasks(vision=True).id)


def test_no_image_decoding(monkeypatch):
    raw = source()
    def fail(*args, **kwargs):
        raise AssertionError('preprocessing must preserve encoded images')
    monkeypatch.setattr(PILImage, 'open', fail)
    ds = template(True, True)(raw)
    instruct = recast_instruct(ds)
    images = instruct['train'].cast_column('images', Sequence(Image(decode=False)))[0]['images']
    assert images == [png('red'), png('blue')]


def test_invalid_gold():
    ds = source()
    ds['train'] = ds['train'].map(lambda x: {'answer': 8})
    with pytest.raises(ValueError, match='label must index'):
        template(True)(ds)


def test_real_single_class_test_split_is_kept():
    ds = source()
    ds['train'] = ds['train'].select([0])
    ds['test'] = ds['validation'].select([1])
    ds = template()(ds)
    assert len(ds['test']) == 1


def test_duplicate_choices_keep_gold_index():
    from tasksource.recast import shuffle_choices
    import random
    result = shuffle_choices({'choice0': 'same', 'choice1': 'same', 'choice2': None, 'labels': 1}, random.Random(1))
    assert result['labels'] == 0
    assert set(result) == {'choice0', 'choice1', 'labels'}


@pytest.mark.parametrize('mc', [False, True])
def test_multimodal_jev(mc):
    from tasksource import recast_jev, render_typed_decision
    from tasksource.preprocess import disable_image_decoding
    task = template(mc, True)
    task.metadata = lambda x: {'source_html': '<button>original</button>'}
    ds = task(source())
    jev = recast_jev(ds, task='test')
    assert jev['train'].features['images'] == Sequence(Image())
    encoded = disable_image_decoding(jev)['train'][0]
    assert encoded['images'] == [png('red'), png('blue')]
    assert encoded['metadata'] == '{"source_html": "<button>original</button>"}'
    assert encoded['answer'] == encoded['criteria'][encoded['label']]
    assert encoded['answer'] == ('gold' if mc else 'no')
    assert render_typed_decision(encoded)['images'] == encoded['images']


def test_image_aware_deduplication():
    from scripts.build_jev_dataset import merge_identical_inputs
    features = Features({'images': Sequence(Image(decode=False)), 'state': Value('string'),
                         'question': Value('string'), 'options': Sequence(Value('string')),
                         'target': Sequence(Value('float64'))})
    rows = Dataset.from_list([
        {'images': [png(color)], 'state': 'Same question', 'question': 'Answer?',
         'options': ['yes', 'no'], 'target': target}
        for color, target in [('red', [1., 0.]), ('blue', [0., 1.]), ('red', [1., 0.])]], features=features)
    result = merge_identical_inputs(rows)
    assert len(result) == 2
    assert result['target'] == [[1., 0.], [0., 1.]]


def test_vision_config_builder(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import scripts.build_jev_dataset as builder
    from tasksource import recast_jev
    encoded = Dataset.from_list([
        {'images': [png(color)], 'inputs': text, 'labels': label}
        for color, label, text in [('red', 0, 'Which color?'), ('blue', 1, 'Which color?'), ('red', 0, 'Name the color.')]], features=Features({
            'images': Sequence(Image()), 'inputs': Value('string'), 'labels': ClassLabel(names=['red', 'blue'])}))
    ds = DatasetDict(train=encoded, validation=encoded.select([0]))
    monkeypatch.setattr(builder, 'load_task', lambda *args, **kwargs: recast_jev(ds))
    monkeypatch.setattr(builder, 'source_licenses', lambda sources: {
        source: {'license': 'unspecified', 'license_use': 'unspecified'} for source in sources})
    args = SimpleNamespace(output=tmp_path, tasks=['nlvr2'], limit=None, max_rows=2,
                           max_rows_eval=1, repo_id='test/repo', upload=False)
    result = builder.build_vision(args)
    assert len(result['train']) == 3
    assert 'validation' not in result  # the source evaluation image is present in training
    assert result['train'].features['images'] == Sequence(Image())
    assert len(set(result['train']['group_id'])) == 3
    assert len(set(result['train']['example_id'])) == 3
    assert len({json.loads(row['metadata'])['image_group_id'] for row in result['train']}) == 2
    for row in result['train']:
        metadata = json.loads(row['metadata'])
        assert metadata['provenance']['revision']
        assert metadata['licenses']['license'] == 'unspecified'
    assert (tmp_path / 'sources.yaml').exists()


def mind2web_source():
    import json
    def option(node, box='50,20,10,10', **flags):
        return json.dumps({'backend_node_id': str(node), 'tag': 'button',
            'attributes': json.dumps({'bounding_box_rect': box, 'aria_label': 'Search'}), **flags})
    buffer = io.BytesIO()
    PILImage.new('RGB', (100, 100), 'red').save(buffer, format='PNG')
    image = {'bytes': buffer.getvalue(), 'path': None}
    row = {'screenshot': image, 'operation': json.dumps({'op': 'CLICK', 'original_op': 'CLICK', 'value': 'SECRET_VALUE'}),
        'pos_candidates': [option('gold', is_original_target=True)],
        'neg_candidates': [option(i, '0,0,10,10') for i in range(30)] + [option('gold')],
        'action_uid': 'action', 'annotation_id': 'train-episode', 'confirmed_task': 'Search for cats',
        'action_reprs': ['PAST', 'SECRET_CURRENT', 'SECRET_FUTURE'], 'target_action_index': '1',
        'target_action_reprs': 'SECRET_TARGET', 'raw_html': 'SECRET_RAW', 'website': 'example', 'domain': 'search'}
    features = Features({'screenshot': Image(), **{k: Value('string') for k, v in row.items() if isinstance(v, str)},
        **{k: Sequence(Value('string')) for k in ('pos_candidates', 'neg_candidates', 'action_reprs')}})
    bad_parent = {**row, 'pos_candidates': [option('gold', is_original_target=False, is_top_level_target=True)]}
    bad_box = {**row, 'pos_candidates': [option('gold', '99,20,10,10', is_original_target=True)]}
    ambiguous = {**row, 'pos_candidates': [option('gold', is_original_target=True), option('other', is_original_target=True)]}
    splits = {'train': Dataset.from_list([row, bad_parent, bad_box, ambiguous], features=features)}
    for split in ('test_task', 'test_website', 'test_domain'):
        splits[split] = Dataset.from_list([{**row, 'annotation_id': split + '-episode'}], features=features)
    return DatasetDict(splits)


def test_grid_coordinates():
    from tasksource.vision_tasks import grid_labels, grid_center
    assert grid_labels((.73, .42)) == [(2, 5)]
    assert grid_labels((0, 0), depth=2) == [(0, 0), (0, 0)]
    assert grid_labels((1, 1), bins=(5, 7), depth=2) == [(6, 4), (6, 4)]
    assert grid_labels((.5, .25), bins=(4, 4)) == [(1, 2)]
    for point in ((0, 0), (1, 1), (.73, .42), (.5, .25)):
        cells = grid_labels(point, (5, 7), depth=2)
        assert grid_labels(grid_center(cells, (5, 7)), (5, 7), depth=2) == cells
    for point in ((-.1, 0), (0, 1.1), (float('nan'), 0), (float('inf'), 0)):
        with pytest.raises(ValueError):
            grid_labels(point)
    for bins, depth in (((0, 7), 1), ((7.5, 7), 1), ((7, 7), 0)):
        with pytest.raises(ValueError):
            grid_labels((0, 0), bins, depth)
    with pytest.raises(ValueError):
        grid_center([(7, 0)])


def test_mind2web_annotations_and_grouping(monkeypatch):
    import json
    import tasksource.access as access
    from tasksource import render_typed_decision_group
    from tasksource.preprocess import disable_image_decoding
    monkeypatch.setattr(access, 'load_dataset', lambda *a, **kw: mind2web_source())
    views = {}
    for name in ('action', 'element', 'x10', 'y10', 'grid5', 'grid7'):
        task = 'mind2web/' + name
        ds = load_task(task, vision=True, max_rows=10, max_rows_eval=10)
        assert set(ds) == {'train', 'test_task', 'test_website', 'test_domain'}
        assert all(len(rows) == 1 for rows in ds.values())
        row = disable_image_decoding(ds)['train'][0]
        assert row['inputs'] == 'Search for cats\nPast actions:\nPAST'
        assert 'SECRET' not in row['inputs']
        metadata = json.loads(row['metadata'])
        assert metadata['point'] == [.55, .25]
        assert metadata['bbox'] == [50., 20., 10., 10.]
        assert metadata['trajectory_id'] == 'train-episode'
        assert all(json.loads(rows[0]['metadata'])['trajectory_id'] != 'train-episode'
                   for split, rows in ds.items() if split != 'train')
        if name == 'element':
            assert len([k for k in row if k.startswith('choice')]) == 24
            assert json.loads(row[f"choice{row['labels']}"])['node'] == 'gold'
            for key, value in row.items():
                if key.startswith('choice'):
                    assert 'is_original_target' not in value and 'SECRET' not in value
        elif name != 'action':
            expected = {'x10': 5, 'y10': 2, 'grid5': 7, 'grid7': 10}[name]
            assert row['labels'] == expected
        jev = load_task(task, vision=True, recast='jev', max_rows=10, max_rows_eval=10)
        views[name] = disable_image_decoding(jev)['train'][0]
        assert views[name]['answer'] == views[name]['criteria'][views[name]['label']]
        assert views[name]['images'] == row['images']
    x, y = views['x10'], views['y10']
    assert x['kind'] == y['kind'] == 'score'
    assert x['group'] == y['group'] and x['source_row'] == y['source_row']
    assert len(views['grid7']['criteria']) == 49
    request = render_typed_decision_group([x, y])
    assert list(request['questions']) == ['x10', 'y10']
    assert request['images'] == x['images']
    with pytest.raises(ValueError, match='ordered images'):
        render_typed_decision_group([x, {**y, 'images': [png('blue')]}])
    with pytest.raises(ValueError, match='source_row'):
        render_typed_decision_group([x, {**y, 'source_row': 'other-action'}])
    with pytest.raises(ValueError, match='Duplicate question'):
        render_typed_decision_group([x, x])


def test_score_only_validation():
    from tasksource import recast_jev
    ds = template()(source())
    with pytest.raises(ValueError, match='ordered label set'):
        recast_jev(ds, score_only=True)
    result = recast_jev(ds, ordinal=True, score_only=True)
    assert set(result['train']['kind']) == {'score'}


def test_gui_builder_keeps_action_groups(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.build_jev_dataset as builder
    import tasksource.access as access
    monkeypatch.setattr(access, 'load_dataset', lambda *a, **kw: mind2web_source())
    monkeypatch.setattr(builder, 'source_licenses', lambda sources: {
        source: {'license': 'openrail', 'license_use': 'unspecified'} for source in sources})
    args = SimpleNamespace(output=tmp_path, tasks=['mind2web/x10', 'mind2web/y10'], limit=None,
                           max_rows=10, max_rows_eval=10, repo_id='test/repo', upload=False)
    ds = builder.build_vision(args)
    assert len(ds['train']) == 2
    assert len(set(ds['train']['group_id'])) == 1
    assert set(ds['train']['question_id']) == {'x10', 'y10'}
    assert len(set(ds['train']['id'])) == 2
    assert set(ds['train']['kind']) == {'score'}
    assert set(ds) == {'train'}  # synthetic evaluation screenshots duplicate training


def test_grouped_recasts_share_text_cleaning():
    import json
    from tasksource import recast_jev
    ds = template()(source())
    ds = ds.map(lambda row: {'inputs': 'Click the A &amp; B button.',
        'metadata': json.dumps({'group': 'shared', 'source_row': 'same-action'})})
    left = recast_jev(ds, task='x10')
    right = recast_jev(ds, task='y10')
    assert left['train']['state'] == right['train']['state']


def test_mind2web_streaming_uses_shared_loader(monkeypatch):
    from datasets import IterableDatasetDict
    import tasksource.access as access
    from tasksource.preprocess import disable_image_decoding
    source = mind2web_source()
    monkeypatch.setattr(access, 'load_dataset', lambda *a, **kw: IterableDatasetDict({
        split: rows.to_iterable_dataset() for split, rows in source.items()}))
    ds = load_task('mind2web/x10', vision=True, recast='jev', streaming=True,
                   max_rows=1, max_rows_eval=1)
    assert set(ds) == set(source)
    assert all(len(rows) == 1 for rows in ds.values())
    assert ds['train'][0]['kind'] == 'score'
    encoded = disable_image_decoding(ds)
    original = source['train'].cast_column('screenshot', Image(decode=False))[0]['screenshot']
    assert encoded['train'][0]['images'] == [original]
