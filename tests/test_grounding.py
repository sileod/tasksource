import io
import json

import pytest
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from PIL import Image as PILImage

from tasksource import load_task, list_tasks
from tasksource.grounding import grounding_row, augment_grounding
from tasksource.preprocess import VisualMultipleChoice
from tasksource.recast import recast_jev
from tasksource.vision_tasks import rico_widget_rows


def image(color="gray"):
    stream = io.BytesIO()
    PILImage.new('RGB', (100, 80), color).save(stream, format='PNG')
    return {'bytes': stream.getvalue(), 'path': None}


def candidates():
    return [{'bbox': [.05, .1, .4, .45], 'text': 'Cancel'},
            {'bbox': [.55, .5, .95, .9], 'text': 'Search'}]


def canonical(gold=1):
    return grounding_row([image(), image()], 'Find search', candidates()[gold]['bbox'],
        candidates=candidates(), metadata={'source_dataset': 'example/gui', 'source_row': 'a',
                                          'image_group_id': 'screen-a'})


def processed(row):
    features = Features({'images': Sequence(Image(decode=False)), 'inputs': Value('string'),
        'labels': Value('int64'), 'choices_list': Sequence(Value('string')), 'metadata': Value('string')})
    source = DatasetDict(train=Dataset.from_list([row], features=features))
    return VisualMultipleChoice(metadata='metadata', choices_list='choices_list')(source)


@pytest.mark.parametrize('variant', ['plain', 'som+text', 'som-only'])
def test_som_alignment_and_target_independence(variant):
    weights = {variant: 1}
    output = augment_grounding(processed(canonical()), weights, seed=7)
    raw = output.cast_column('images', Sequence(Image(decode=False)))['train'][0]
    repeat = augment_grounding(processed(canonical()), weights, seed=7)
    assert raw == repeat.cast_column('images', Sequence(Image(decode=False)))['train'][0]
    other_gold = augment_grounding(processed(canonical(0)), weights, seed=7)
    assert raw['images'] == other_gold.cast_column('images', Sequence(Image(decode=False)))['train'][0]['images']
    assert raw['images'][1] == image()
    if variant == 'plain':
        assert raw['images'][0] == image()
    else:
        with PILImage.open(io.BytesIO(raw['images'][0]['bytes'])) as marked:
            assert marked.getpixel((5, 30)) == (255, 0, 0)
            assert marked.getpixel((55, 65)) == (255, 0, 0)
    recast = recast_jev(output, task='example', question='Select element')['train'][0]
    gold = canonical()['choices_list'][1] if variant == 'plain' else ('Mark 2' if variant == 'som-only' else 'Mark 2: ' + canonical()['choices_list'][1])
    assert recast['answer'] == gold
    assert recast['criteria'][recast['label']] == gold
    assert recast['state'] == 'Find search'
    assert 'target_bbox' not in recast['state']
    assert json.loads(recast['metadata'])['augmentation']['seed'] == 7


def test_invalid_candidates_and_projection():
    with pytest.raises(ValueError, match='Target must match'):
        grounding_row([image()], 'Find', [.1, .1, .2, .2], candidates=candidates(), metadata={})
    with pytest.raises(ValueError, match='unique'):
        grounding_row([image()], 'Find', candidates()[0]['bbox'], candidates=[candidates()[0]]*2, metadata={})
    row = grounding_row([image()], 'Find', [.8, .8, 1, 1], metadata={}, bins=(5, 7))
    assert row['labels'] == 6*5+4
    with pytest.raises(ValueError, match='sum to one'):
        augment_grounding(processed(canonical()), {'plain': .3})


def test_source_exclusion_before_loading(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('Source was loaded despite exclusion')
    monkeypatch.setattr('tasksource.access.load_dataset', fail)
    with pytest.raises(ValueError, match='Excluded source'):
        load_task('mind2web/element', vision=True, excluded_sources=['tasksource/multimodal-mind2web'])
    assert not any('mind2web' in id for id in list_tasks(vision=True, excluded=['mind2web']).id)


def rico_source():
    root = {'bounds': [0, 0, 100, 80], 'children': [
        {'bounds': [5, 8, 40, 36], 'componentLabel': 'Button', 'text': 'Cancel'},
        {'bounds': [55, 40, 95, 72], 'componentLabel': 'Button', 'text': 'Search'}]}
    row = {'screenId': 42, 'image': image(), 'bbox': [.55, .5, .95, .9],
           'captions': ['Find search'], 'semantic_annotations': json.dumps(root)}
    features = Features({'screenId': Value('int64'), 'image': Image(decode=False),
        'bbox': Sequence(Value('float64')), 'captions': Sequence(Value('string')),
        'semantic_annotations': Value('string')})
    return DatasetDict({split: Dataset.from_list([{**row, 'screenId': index, 'image': image(color)}], features=features)
                        for index, (split, color) in enumerate([('train', 'gray'), ('val', 'blue'), ('test', 'green')], 42)})


def test_load_task_som_jev_native_candidates(monkeypatch):
    monkeypatch.setattr('tasksource.access.load_dataset', lambda *args, **kwargs: rico_source())
    ds = load_task('rico-widget/element', vision=True, recast='jev',
                   grounding={'probabilities': {'som-only': 1}}, seed=9)
    assert set(ds) == {'train', 'validation', 'test'}
    row = ds['train'][0]
    assert row['answer'] == 'Mark 2'
    assert row['criteria'][row['label']] == 'Mark 2'
    assert len(row['images']) == 1
    metadata = json.loads(row['metadata'])
    assert metadata['screenId'] == 42
    assert metadata['source_revision']
    assert metadata['candidate_boxes'] == [c['bbox'] for c in candidates()]


def test_rico_missing_target_not_inserted():
    source = rico_source()
    source['train'] = source['train'].map(lambda row: {'bbox': [.1, .2, .3, .4]})
    assert len(rico_widget_rows(source, elements=True)['train']) == 0


def test_split_leakage_detected_before_drawing():
    ds = processed(canonical())
    ds['validation'] = ds['train']
    with pytest.raises(ValueError, match='crosses splits'):
        augment_grounding(ds)


def test_original_source_exclusion():
    catalog = list_tasks(vision=True, excluded_sources=['osunlp/Multimodal-Mind2Web'])
    assert not any('mind2web' in id for id in catalog.id)


def test_label_geometry_mismatch_rejected():
    row = canonical()
    row['labels'] = 0
    with pytest.raises(ValueError, match='Gold label'):
        augment_grounding(processed(row))


def test_crop_stage_preserved():
    row = grounding_row([image()], 'Find widget in crop', [.1, .1, .3, .3],
                        metadata={'stage': 2, 'original_target_bbox': [.4, .4, .5, .5]})
    metadata = json.loads(row['metadata'])
    assert metadata['stage'] == 2
    assert metadata['target_bbox'] == [.1, .1, .3, .3]
    assert metadata['original_target_bbox'] == [.4, .4, .5, .5]


def test_augmented_image_leakage_keeps_original_hash():
    from scripts.build_jev_dataset import image_keys
    row = augment_grounding(processed(canonical())).cast_column('images', Sequence(Image(decode=False)))['train'][0]
    original = json.loads(row['metadata'])['augmentation']['original_image_sha256']
    assert original in image_keys(row['images'], row['metadata'])
