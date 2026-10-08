import io
import os
import pytest
from PIL import Image as PILImage
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value, ClassLabel
from tasksource import list_tasks, load_task, task_provenance, hub_datasets
from tasksource.preprocess import VisualClassification, VisualMultipleChoice, Classification
from tasksource.recast import recast_instruct
from tasksource.vision_tasks import figureqa_rows, grouped_mc_rows
from scripts.repackage_dataset.vision import prepare_cauldron, prepare_mind2web
from tasksource.preprocess import disable_image_decoding


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
    assert list(vision.id) == ['nlvr2', 'aokvqa', 'scienceqa-img', 'ai2d', 'figureqa',
                             'mind2web/action', 'mind2web/element', 'mind2web/x10',
                             'mind2web/y10', 'mind2web/grid5', 'mind2web/grid7',
                             'm3cot', 'exams-v', 'visualsphinx', 'muslr/tfu', 'muslr/mc', 'iconqa/text', 'view2space/mcq', 'visual7w', 'clevr/yesno', 'mapqa/yesno',
                             'tqa', 'hateful-memes', 'clevr/color', 'clevr/shape', 'clevr/size',
                             'clevr/material', 'intergps', 'clevr/count', 'tallyqa/count', 'vsr/yesno',
                             'rico-widget/grid7', 'rico-widget/element']
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
    ds = VisualMultipleChoice(choices_list='choices_list', pre_process=grouped_mc_rows)(
        prepare_cauldron(disable_image_decoding(DatasetDict(train=ai)), 'ai2d'))
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
    result = shuffle_choices({'choice0': 'same', 'choice1': 'same', 'choice2': None, 'labels': 1}, random.Random(1), visual=True)
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
    splits = {'train': Dataset.from_list([row, bad_parent, bad_box, ambiguous, {**row, 'screenshot': None},
        {**row, 'screenshot': {'bytes': b'invalid image', 'path': None}}], features=features)}
    for split in ('test_task', 'test_website', 'test_domain'):
        splits[split] = Dataset.from_list([{**row, 'annotation_id': split + '-episode'}], features=features)
    return DatasetDict(splits)


def mind2web_mirror():
    return prepare_mind2web(disable_image_decoding(mind2web_source()))


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
    monkeypatch.setattr(access, 'load_dataset', lambda *a, **kw: mind2web_mirror())
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
    monkeypatch.setattr(access, 'load_dataset', lambda *a, **kw: mind2web_mirror())
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
    source = mind2web_mirror()
    monkeypatch.setattr(access, 'load_dataset', lambda *a, **kw: IterableDatasetDict({
        split: rows.to_iterable_dataset() for split, rows in source.items()}))
    ds = load_task('mind2web/x10', vision=True, recast='jev', streaming=True,
                   max_rows=1, max_rows_eval=1)
    assert set(ds) == set(source)
    assert all(len(rows) == 1 for rows in ds.values())
    assert ds['train'][0]['kind'] == 'score'
    encoded = disable_image_decoding(ds)
    original = source['train'].cast_column('images', Sequence(Image(decode=False)))[0]['images']
    assert encoded['train'][0]['images'] == original


@pytest.mark.parametrize('task_id', ['m3cot', 'exams-v', 'visualsphinx', 'muslr/tfu', 'muslr/mc', 'iconqa/text'])
def test_reasoning_catalog_mappings(task_id, monkeypatch):
    import json
    import tasksource.access as access
    from tasksource.preprocess import disable_image_decoding
    if task_id == 'm3cot':
        rows = [{'image': png('red'), 'context': 'Context', 'question': 'Question',
                 'choices': ['a', 'b', 'c', 'd', 'gold'], 'answer': 'E', 'id': 'q', 'image_id': 'i',
                 'rationale': 'SECRET_REASON', 'domain': 'science', 'topic': 'math'}]
        features = Features({'image': Image(), 'choices': Sequence(Value('string')),
                             **{k: Value('string') for k, v in rows[0].items() if isinstance(v, str)}})
        source_rows = Dataset.from_list(rows + [{**rows[0], 'image': None}], features=features)
        gold = 'gold'
    elif task_id == 'exams-v':
        source_rows = Dataset.from_list([{'image': png('red'), 'answer_key': 'D', 'sample_id': 'q',
            'language': 'Arabic', 'subject': 'Chemistry', 'grade': '10'}], features=Features({
                'image': Image(), **{k: Value('string') for k in ('answer_key', 'sample_id', 'language', 'subject', 'grade')}}))
        gold = 'fourth option'
    elif task_id == 'visualsphinx':
        source_rows = Dataset.from_list([{'images': [png('red'), png('blue')], 'problem': '<image>Question',
            'choice': json.dumps({k: k for k in 'ABCDEFGHIJ'}), 'answer': 'J', 'id': 'q',
            'explanation': 'SECRET_REASON', 'readability': 4, 'reasonableness': 4, 'has_duplicate': False}],
            features=Features({'images': Sequence(Image()), **{k: Value('string') for k in ('problem', 'choice', 'answer', 'id', 'explanation')},
                'readability': Value('int64'), 'reasonableness': Value('int64'), 'has_duplicate': Value('bool')}))
        gold = 'J'
    elif task_id.startswith('muslr/'):
        common = {'image': png('red'), 'full_context': 'Context', 'question': 'Question', 'id': 'q',
                  'domain': 'science', 'symbol': 'pl', 'depth': 2, 'reasoning': 'SECRET_REASON'}
        source_rows = Dataset.from_list([{**common, 'choices': None, 'answer': 'Unknown'},
            {**common, 'choices': json.dumps(['A. wrong', 'B. gold']), 'answer': 'B'}],
            features=Features({'image': Image(), 'depth': Value('int64'),
                **{k: Value('string') for k in ('full_context', 'question', 'id', 'domain', 'symbol', 'reasoning', 'choices', 'answer')}}))
        gold = 'Unknown' if task_id.endswith('tfu') else 'gold'
    else:
        source_rows = Dataset.from_list([{'images': [png('red')], 'texts': [
            {'user': 'Question: Q\nChoices:\nA. wrong\nB. gold\nAnswer with the letter.', 'assistant': 'Answer: B', 'source': 'IconQA'},
            {'user': 'Question: Open answer?', 'assistant': 'SECRET_REASON', 'source': 'IconQA'}]}],
            features=Features({'images': Sequence(Image()), 'texts': [{'user': Value('string'), 'assistant': Value('string'), 'source': Value('string')}]}))
        gold = 'gold'
    source_ds = DatasetDict(train=source_rows)
    if task_id == 'iconqa/text':
        source_ds = prepare_cauldron(disable_image_decoding(source_ds), 'iconqa')
    monkeypatch.setattr(access, 'load_dataset', lambda *a, **kw: source_ds)
    ds = load_task(task_id, vision=True, recast='jev')
    assert set(ds) == {'train'} and len(ds['train']) == 1
    row = disable_image_decoding(ds)['train'][0]
    assert row['answer'] == gold == row['criteria'][row['label']]
    assert 'SECRET_REASON' not in row['state']
    assert row['images'][0] == png('red')
    if task_id == 'visualsphinx':
        assert len(row['criteria']) == 10
        assert row['images'] == [png('red'), png('blue')]
    if task_id == 'm3cot':
        assert len(row['criteria']) == 5
        assert json.loads(row['metadata'])['rationale'] == 'SECRET_REASON'


def test_sampled_image_paths_embed_original_bytes(monkeypatch):
    import datasets.utils.file_utils as file_utils
    from tasksource.preprocess import disable_image_decoding
    raw = Dataset.from_list([{'images': [{'path': f'https://example.com/{i}.png', 'bytes': None}], 'inputs': str(i), 'labels': 'yes'}
        for i in range(5)], features=Features({'images': Sequence(Image()), 'inputs': Value('string'), 'labels': Value('string')}))
    calls = []
    def reader(path, mode):
        calls.append(path)
        return io.BytesIO(png('red')['bytes'])
    monkeypatch.setattr(file_utils, 'xopen', reader)
    ds = VisualClassification()(DatasetDict(train=raw), max_rows=2)
    assert len(calls) == 2
    assert all(row['images'] == [png('red')] for row in disable_image_decoding(ds)['train'])


def test_view2space_repackaging(tmp_path, monkeypatch):
    import json
    import zipfile
    import huggingface_hub
    from scripts.upload_repackaged import view2space
    monkeypatch.chdir(tmp_path)
    rows = [{'q_idx': str(i), 'q_type': 'mcq', 'question': 'Question', 'question_prompt': 'Prompt',
        'options': {'A': 'wrong', 'B': 'gold'}, 'answer': 'B', 'image_paths': ['images/red.png', 'images/blue.png'],
        'supporting': {'chain_of_thought': 'SECRET_REASON', 'draw_boxes': None}} for i in range(2)]
    rows.append({**rows[0], 'q_type': 'count', 'options': {}, 'answer': 2})
    rows.append({**rows[0], 'q_idx': 'dedup', 'options': {'A': 'wrong', 'B': 'wrong', 'C': 'gold'}, 'answer': 'C'})
    rows.append({**rows[0], 'q_idx': 'ambiguous', 'options': {'A': 'gold', 'B': 'gold'}, 'answer': 'A'})
    (tmp_path / 'overall.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    with zipfile.ZipFile(tmp_path / 'images.zip', 'w') as archive:
        for color in ('red', 'blue'):
            archive.writestr('home/source/images/' + color + '.png', png(color)['bytes'])
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda repo, filename, **kw: str(tmp_path / filename))
    directory = view2space()
    from datasets import load_dataset
    from tasksource.preprocess import disable_image_decoding
    ds = disable_image_decoding(load_dataset(str(directory), download_mode='force_redownload'))
    assert set(ds) == {'train'} and len(ds['train']) == 1
    row = ds['train'][0]
    from datasets.utils.file_utils import xopen
    assert [xopen(image['path'], 'rb').read() for image in row['images']] == [png('red')['bytes'], png('blue')['bytes']]
    assert len(row['qa']) == 3
    assert set(row['qa'][0]) == {'inputs', 'choices_list', 'labels', 'metadata'}
    assert row['qa'][0]['choices_list'][row['qa'][0]['labels']] == 'gold'
    assert 'SECRET_REASON' not in row['qa'][0]['inputs']
    assert json.loads(row['qa'][0]['metadata'])['reasoning'] == 'SECRET_REASON'
    from tasksource.vision_tasks import grouped_mc_rows
    flat = VisualMultipleChoice(choices_list='choices_list', metadata='metadata', pre_process=grouped_mc_rows)(ds)
    assert len(flat['train']) == 3
    assert flat['train'][0]['choice1'] == 'gold' and flat['train'][0]['labels'] == 1
    assert flat['train'][2]['choice1'] == 'gold' and flat['train'][2]['labels'] == 1
    assert json.loads(flat['train'][2]['metadata'])['option_keys'] == ['A', 'C']
    report = json.loads((directory / 'provenance.json').read_text())
    assert report['questions'] == 3 and report['excluded_questions'] == 1
    assert report['deduplicated_distractor_rows'] == 1


def test_uniform_streaming_sample_covers_the_entire_source():
    from tasksource.preprocess import reservoir_sample
    seen = []
    def rows():
        for i in range(10000):
            seen.append(i)
            yield {'index': i}
    selected = reservoir_sample(rows(), 100, seed=0)
    assert len(seen) == 10000 and len(selected) == 100
    assert selected == reservoir_sample(({'index': i} for i in range(10000)), 100, seed=0)
    indices = [row['index'] for row in selected]
    assert indices == sorted(indices) and len(set(indices)) == 100
    assert sum(i >= 5000 for i in indices) > 30
    assert max(indices) > 9900
    assert reservoir_sample(iter([1, 2]), 10) == [1, 2]


def test_text_shuffling_retains_its_original_option_slots():
    from tasksource.recast import shuffle_choices
    import random
    result = shuffle_choices({'choice0': 'same', 'choice1': 'same', 'choice2': None, 'labels': 1}, random.Random(1))
    assert result == {'choice0': 'same', 'choice1': None, 'choice2': 'same', 'labels': 0}


@pytest.mark.parametrize('answer, position', [('A', 0), ('b', 1), ('E', 4), ('1', 0), ('4', 3),
                                            ('А', 0), ('Б', 1), ('В', 2), ('г', 3), ('Д', 4),
                                            ('-1', None), (None, None), ('bad', None)])
def test_exam_option_positions(answer, position):
    from tasksource.vision_tasks import exam_option_position
    assert exam_option_position(answer) == position


def test_snli_ve_is_parked_for_label_quality():
    from tasksource import parked
    assert parked.PARKED['snli_ve'][0] == 'unsound'
    assert 'snli-ve' not in set(list_tasks(vision=True).id)
    assert 'training source is uncorrected' in parked.REASONS['snli_ve']


def test_mind2web_repackaging_audit(tmp_path, monkeypatch):
    import json
    import scripts.repackage_dataset.vision as conversion
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(conversion, '_parquet_source', lambda *a, **kw: disable_image_decoding(mind2web_source()))
    ds = conversion.mind2web()
    assert set(ds) == {'train', 'test_task', 'test_website', 'test_domain'}
    assert all(len(rows) == 1 for rows in ds.values())
    assert ds['train'][0]['labels'] < len(ds['train'][0]['choices_list'])
    directory = tmp_path / 'build/multimodal-mind2web-release'
    report = json.loads((directory / 'provenance.json').read_text())
    assert report['excluded_rows'] == 5
    assert report['splits']['train']['questions'] == 1
    assert report['conversion_sha256'] and report['revision']
    assert len((directory / 'excluded-questions.jsonl').read_text().splitlines()) == 5
    # Rebuilding must run eligibility auditing even when dataset caches exist.
    conversion.mind2web()
    assert json.loads((directory / 'provenance.json').read_text())['excluded_rows'] == 5


from tasksource.vision_tasks import cauldron_mc_rows, cauldron_closed_rows, parse_cauldron_mc


def cauldron_source(answers=('Answer: B',), questions=None):
    question = 'Question: Which?\nChoices:\nA. first\nB. second\nAnswer with the letter.'
    return DatasetDict(train=Dataset.from_list([
        {'images': [png('red'), png('blue')], 'texts': [
            {'user': q, 'assistant': a, 'source': 'native'}
            for q, a in zip(questions or [question] * len(answers), answers)]}],
        features=Features({'images': Sequence(Image(decode=False)), 'texts': [
            {'user': Value('string'), 'assistant': Value('string'), 'source': Value('string')}]})))


@pytest.mark.parametrize('question,answer,reason', [
    ('Question: open', 'Answer: A', 'no_explicit_choices'),
    ('Question: Q\nChoices:\nA. same\nB. SAME\nAnswer with the letter.', 'Answer: A', 'duplicate_options'),
    ('Question: Q\nChoices:\nA. a\nC. c\nAnswer with the letter.', 'Answer: A', 'invalid_option_format'),
    ('Question: Q\nChoices:\nA. a\nB. b\nAnswer with the letter.', 'Answer: C', 'invalid_gold'),
])
def test_cauldron_invalid_mc(question, answer, reason):
    record, why = parse_cauldron_mc({'user': question, 'assistant': answer})
    assert record is None and why == reason


def test_cauldron_grouped_images_and_native_gold():
    import json
    ds = cauldron_mc_rows(cauldron_source(('Answer: B', 'Answer: A')))
    assert len(ds['train']) == 2
    assert ds['train']['labels'] == [1, 0]
    assert ds['train'][0]['images'] == [png('red'), png('blue')]
    metadata = [json.loads(value) for value in ds['train']['metadata']]
    assert metadata[0]['image_group_id'] == metadata[1]['image_group_id']
    assert metadata[0]['id'] != metadata[1]['id']
    assert metadata[0]['source_answer'] == 'Answer: B'
    from datasets import IterableDatasetDict
    streamed = cauldron_mc_rows(IterableDatasetDict(train=cauldron_source()['train'].to_iterable_dataset()))
    assert next(iter(streamed['train']))['labels'] == 1


@pytest.mark.parametrize('task', ['visual7w', 'clevr/yesno', 'mapqa/yesno', 'tqa', 'hateful-memes',
                                 'clevr/color', 'clevr/shape', 'clevr/size', 'clevr/material', 'intergps',
                                 'clevr/count', 'tallyqa/count', 'vsr/yesno'])
def test_cauldron_task_recasts(task, monkeypatch):
    import json
    import tasksource.access as access
    mapping = list_tasks(vision=True).set_index('id').loc[task, 'mapping']
    answer = ('Answer: B' if isinstance(mapping, VisualMultipleChoice)
              else next(iter(mapping.label_values)).capitalize() + '.')
    native = cauldron_source((answer, 'not in vocabulary'))
    monkeypatch.setattr(access, 'load_dataset', lambda *args, **kwargs: native)
    ds = load_task(task, vision=True, max_rows=10)
    assert set(ds) == {'train'} and len(ds['train']) == 1
    assert isinstance(ds['train'].features['labels'], ClassLabel)
    assert ds['train'].features['images'] == Sequence(Image())
    result = load_task(task, vision=True, max_rows=10, recast='jev')
    row = disable_image_decoding(result)['train'][0]
    assert row['images'] == [png('red'), png('blue')]
    assert row['answer'] == row['criteria'][row['label']]
    metadata = json.loads(row['metadata'])
    assert metadata['image_group_id']
    if task == 'hateful-memes':
        assert metadata['source_redistribution'] == 'restricted'
        assert 'LICENSE.txt' in metadata['source_license_url']


def test_cauldron_closed_vocabulary():
    ds = cauldron_closed_rows(cauldron_source(('Yes.', 'Answer: NO.', 'Blue.', 'Unknown.')), ('yes', 'no'))
    assert ds['train']['labels'] == ['yes', 'no']


def test_vision_publish_rejects_restricted_images(tmp_path, monkeypatch):
    import scripts.build_jev_dataset as builder
    ds = DatasetDict(train=Dataset.from_list([{'source': 'vision/hateful-memes',
        'metadata': '{"source_redistribution": "restricted"}'}]))
    calls = []
    monkeypatch.setattr(DatasetDict, 'push_to_hub', lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match='hateful-memes'):
        builder.publish_vision(ds, tmp_path, 'test/repo')
    assert not calls


def test_rico_widget_captions_geometry_and_recast(monkeypatch):
    import json
    import tasksource.access as access
    features = Features({'screenId': Value('int64'), 'image': Image(),
        'bbox': Sequence(Value('float64')), 'captions': Sequence(Value('string'))})
    def rows(screen):
        return Dataset.from_list([
            {'screenId': screen, 'image': png('red'), 'bbox': [0.8, 0.8, 1.0, 1.0],
             'captions': ['open settings', 'settings menu', '']},
            {'screenId': screen, 'image': png('blue'), 'bbox': [0.5, 0.5, 0.2, 0.2],
             'captions': ['invalid box']}], features=features)
    native = DatasetDict(train=rows(1), val=rows(2), test=rows(3))
    monkeypatch.setattr(access, 'load_dataset', lambda *args, **kwargs: native)
    ds = load_task('rico-widget/grid7', vision=True, recast='jev', max_rows=10)
    assert set(ds) == {'train', 'validation', 'test'}
    for split in disable_image_decoding(ds).values():
        assert len(split) == 2
        raw = split
        assert raw[0]['images'] == [png('red')]
        assert all(row['answer'] == 'r6c6' for row in raw)
        metadata = [json.loads(row['metadata']) for row in raw]
        assert metadata[0]['source_widget'] == metadata[1]['source_widget']
        assert metadata[0]['source_row'] != metadata[1]['source_row']
        assert metadata[0]['target_bbox'] == [0.8, 0.8, 1.0, 1.0]


@pytest.mark.parametrize('task,maximum', [('clevr/count', 10), ('tallyqa/count', 15)])
def test_visual_counts_keep_full_numeric_ontology(task, maximum, monkeypatch):
    import tasksource.access as access
    native = cauldron_source((f'{maximum}.', '0.', str(maximum + 1)))
    monkeypatch.setattr(access, 'load_dataset', lambda *args, **kwargs: native)
    canonical = load_task(task, vision=True, max_rows=10)
    assert canonical['train'].features['labels'].names == [str(i) for i in range(maximum + 1)]
    assert set(canonical['train']['labels']) == {0, maximum}
    recast = load_task(task, vision=True, recast='jev', max_rows=10)
    assert set(recast['train']['answer']) == {'0', str(maximum)}
    assert all(row['kind'] == 'choice' and len(row['criteria']) == maximum + 1
               for row in recast['train'])


def test_vision_builder_distinguishes_option_permutations(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.build_jev_dataset as builder
    ds = DatasetDict(train=Dataset.from_list([
        {'images': [png('red')], 'state': 'Same question', 'instructions': 'Choose.',
         'criteria': options, 'label': label}
        for options, label in [(['yes', 'no'], 0), (['no', 'yes'], 1)]], features=Features({
            'images': Sequence(Image(decode=False)), 'state': Value('string'),
            'instructions': Value('string'), 'criteria': Sequence(Value('string')), 'label': Value('int64')})))
    monkeypatch.setattr(builder, 'load_task', lambda *args, **kwargs: ds)
    monkeypatch.setattr(builder, 'source_licenses', lambda sources: {
        source: {'license': 'unspecified', 'license_use': 'unspecified'} for source in sources})
    args = SimpleNamespace(output=tmp_path, tasks=['nlvr2'], limit=None, max_rows=2,
                           max_rows_eval=1, repo_id='test/repo', upload=False)
    result = builder.build_vision(args)['train']
    assert len(result) == len(set(result['id'])) == 2
    assert len(set(result['group_id'])) == 1
    assert all(row['options'][row['target'].index(1.)] == 'yes' for row in result)
