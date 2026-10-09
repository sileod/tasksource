"""Offline checks of rendered supervision, image roles and source selection."""
import io
import json
import tarfile

import numpy as np
import pytest
from PIL import Image as PILImage
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Sequence, Value

from scripts.repackage_dataset.regions import (
    bapps_label, correspondence_example, digest, pixel_box, region_example, render_region, selected,
)
from tasksource.preprocess import VisualClassification, VisualMultipleChoice
from tasksource.recast import recast_jev


def encoded(color='white', size=(100, 100)):
    buffer = io.BytesIO()
    PILImage.new('RGB', size, color).save(buffer, format='PNG')
    return buffer.getvalue()


def test_region_marks_geometry_without_encoding_class():
    source = encoded()
    first = region_example(source, [20, 25, 30, 35], 0, {'source_row': 'a'})
    second = region_example(source, [20, 25, 30, 35], 1, {'source_row': 'b'})
    assert first['images'] == second['images']
    assert first['inputs'] == second['inputs']
    marked = np.asarray(PILImage.open(io.BytesIO(first['images'][0]['bytes'])))
    assert tuple(marked[25, 20]) == (255, 0, 0)
    assert tuple(marked[40, 35]) == (255, 255, 255)
    assert tuple(marked[0, 0]) == (255, 255, 255)
    info = json.loads(first['metadata'])['augmentation']
    assert info['original_image_sha256'] == digest(source)
    assert info['box_xyxy'] == [20, 25, 50, 60]
    assert info['resize'] is None


def test_native_segment_mask_and_box_must_agree():
    source = encoded()
    mask = PILImage.new('L', (100, 100))
    pixels = np.zeros((100, 100), dtype=np.uint8)
    pixels[20:50, 10:40] = 255
    mask = PILImage.fromarray(pixels)
    marked, info = render_region(source, [10, 20, 30, 30], mask=mask)
    image = np.asarray(PILImage.open(io.BytesIO(marked)))
    assert image[30, 25, 0] == 255
    assert image[30, 25, 1] < 255
    assert tuple(image[70, 70]) == (255, 255, 255)
    assert info['mask_sha256'] == digest(mask.tobytes())
    with pytest.raises(ValueError, match='mask_box_mismatch'):
        render_region(source, [50, 50, 20, 20], mask=mask)
    with pytest.raises(ValueError, match='out_of_frame_box'):
        pixel_box([-1, 0, 10, 10], (100, 100))


def test_large_region_ontology_survives_preprocessing_and_jev():
    rows = [region_example(encoded(), [20, 20, 30, 30], label, {'source_row': str(label)})
            for label in (0, 1202)]
    names = [f'category {i}' for i in range(1203)]
    features = Features({'images': Sequence(Image()), 'inputs': Value('string'),
                         'labels': ClassLabel(names=names), 'metadata': Value('string')})
    data = DatasetDict(train=Dataset.from_list(rows, features=features))
    task = VisualClassification(metadata='metadata', question='What is the highlighted category?')
    prepared = task(data)
    assert prepared['train'].features['labels'].names == names
    decisions = recast_jev(prepared)['train']
    for row, label in zip(decisions, (0, 1202)):
        assert row['answer'] == names[label]
        assert len(row['criteria']) == 1203
        assert row['images'][0].size == (100, 100)
        assert json.loads(row['metadata'])['augmentation']['original_image_sha256'] == digest(encoded())


def test_bapps_vote_direction_ties_and_image_roles_survive_permutation():
    assert bapps_label(0) == 0
    assert bapps_label(.2) == 0
    assert bapps_label(.5) is None
    assert bapps_label(.6) == 1
    assert bapps_label(1) == 1
    for value in (float('nan'), -1, 2):
        with pytest.raises(ValueError):
            bapps_label(value)
    images = [{'bytes': encoded(color), 'path': None} for color in ('white', 'red', 'blue')]
    source = DatasetDict(train=Dataset.from_list([{'images': images, 'inputs': 'Compare to image 1.',
        'choices_list': ['Image 2', 'Image 3'], 'labels': 1}], features=Features({
            'images': Sequence(Image()), 'inputs': Value('string'),
            'choices_list': Sequence(Value('string')), 'labels': Value('int64')})))
    prepared = VisualMultipleChoice(choices_list='choices_list')(source)
    row = recast_jev(prepared, task='bapps/preference')['train'][0]
    assert row['criteria'][row['label']] == 'Image 3'
    assert [image.getpixel((0, 0)) for image in row['images']] == [
        (255, 255, 255), (255, 0, 0), (0, 0, 255)]


def test_correspondence_marks_only_source_and_uses_target_dimensions():
    source, target = encoded(size=(100, 100)), encoded('blue', (200, 100))
    row = correspondence_example(source, target, [20, 30], [199, 99], {'source_row': 'pair:0'})
    assert row['images'][1]['bytes'] == target
    assert row['labels'] == 48
    assert tuple(np.asarray(PILImage.open(io.BytesIO(row['images'][0]['bytes'])))[30, 20]) == (255, 0, 0)
    info = json.loads(row['metadata'])
    assert info['target_image_size'] == [200, 100]
    assert info['target_point'] == [199, 99]
    assert '199' not in row['inputs'] and '99' not in row['inputs']
    with pytest.raises(ValueError, match='out_of_frame_keypoint'):
        correspondence_example(source, target, [20, 30], [200, 99], {})


def test_bounded_source_selection_is_order_independent_and_not_a_prefix():
    rows = [{'source_row': str(i), 'labels': i % 2} for i in range(100)]
    first = selected(rows, 10, 42)
    assert first == selected(reversed(rows), 10, 42)
    assert first != rows[:10]
    relabeled = [{**row, 'labels': 1-row['labels']} for row in rows]
    assert [row['source_row'] for row in first] == [row['source_row'] for row in selected(relabeled, 10, 42)]


def test_bapps_archive_preserves_native_votes_and_excludes_shared_references(tmp_path, monkeypatch):
    import scripts.repackage_dataset.regions as conversion
    archives = {}
    for split, cases in [('train', [('a', 'red', 0.), ('tie', 'white', .5)]),
                         ('val', [('overlap', 'red', 1.), ('b', 'blue', .8)])]:
        path = tmp_path / (split + '.tar')
        with tarfile.open(path, 'w') as archive:
            for name, color, vote in cases:
                buffer = io.BytesIO(); np.save(buffer, np.array([vote], dtype=np.float32))
                files = {'judge': buffer.getvalue(), 'ref': encoded(color),
                         'p0': encoded('white'), 'p1': encoded('black')}
                for folder, data in files.items():
                    member = tarfile.TarInfo(f'{split}/traditional/{folder}/{name}.' + ('npy' if folder == 'judge' else 'png'))
                    member.size = len(data); archive.addfile(member, io.BytesIO(data))
        archives['twoafc_' + split + '.tar.gz'] = path
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'build').mkdir()
    monkeypatch.setattr(conversion, 'archive_file', archives.__getitem__)
    result = conversion.bapps(max_rows=10, max_rows_eval=10, seed=42)
    assert {split: len(rows) for split, rows in result.items()} == {'train': 1, 'validation': 1}
    assert result['train'][0]['labels'] == 0 and result['validation'][0]['labels'] == 1
    assert result['validation'][0]['choices_list'] == ['Image 2', 'Image 3']
    info = json.loads(result['validation'][0]['metadata'])
    assert info['annotators'] == 5 and info['human_preference_p1'] == pytest.approx(.8)
    assert info['image_roles'] == ['reference', 'p0', 'p1']
    report = json.loads((tmp_path / 'build/bapps-release/provenance.json').read_text())
    assert report['excluded_by_reason'] == {'tied_human_votes': 1, 'reference_image_crosses_splits': 1}


@pytest.mark.parametrize('task_id,classes,image_count', [
    ('lvis/region', 1203, 1), ('coco/panoptic-region', 133, 1),
    ('doclaynet/region', 11, 1), ('bapps/preference', 2, 3), ('spair71k/grid7', 49, 2),
])
def test_catalog_loader_and_jev_use_prepared_fields(task_id, classes, image_count, monkeypatch):
    from tasksource import load_task, list_tasks
    import tasksource.access as access
    images = [{'bytes': encoded(color), 'path': None} for color in ('red', 'blue', 'white')[:image_count]]
    row = {'images': images, 'inputs': 'Locate or identify the displayed target.',
           'labels': classes-1, 'metadata': json.dumps({'source_row': 'native:1', 'bbox': [1, 2, 3, 4]})}
    features = Features({'images': Sequence(Image()), 'inputs': Value('string'),
        'labels': ClassLabel(names=[f'class {i}' for i in range(classes)]), 'metadata': Value('string')})
    if task_id == 'bapps/preference':
        row['choices_list'] = ['Image 2', 'Image 3']
        features['choices_list'] = Sequence(Value('string'))
        features['labels'] = Value('int64')
    source = DatasetDict(train=Dataset.from_list([row], features=features))
    monkeypatch.setattr(access, 'load_dataset', lambda *args, **kwargs: source)
    canonical = load_task(task_id, vision=True, max_rows=1)
    assert set(canonical) == {'train'}
    assert len(canonical['train'].features['labels'].names) == classes
    result = load_task(task_id, vision=True, max_rows=1, recast='jev')['train'][0]
    assert len(result['images']) == image_count
    assert result['criteria'][result['label']] == ('Image 3' if task_id == 'bapps/preference' else f'class {classes-1}')
    assert json.loads(result['metadata'])['source_row'] == 'native:1'
    catalog = list_tasks(vision=True, excluded_sources=[
        list_tasks(vision=True).set_index('id').loc[task_id, 'dataset_name']])
    assert task_id not in set(catalog.id)


def test_region_license_filter_preserves_image_terms_and_uncertainty():
    from tasksource import task_licenses
    uses = task_licenses(['lvis/region', 'coco/panoptic-region', 'doclaynet/region',
                         'bapps/preference', 'spair71k/grid7'], vision=True).set_index('id').license_use
    assert uses['lvis/region'] == uses['coco/panoptic-region'] == 'non-commercial'
    assert uses['doclaynet/region'] == 'commercial'
    assert uses['bapps/preference'] == uses['spair71k/grid7'] == 'unspecified'
