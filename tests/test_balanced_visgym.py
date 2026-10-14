import io
import json
from collections import Counter

import pytest
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from scripts.repackage_dataset.procedural.balanced_visgym import FAMILIES, family_selection, visgym_balanced
from scripts.repackage_dataset.procedural.common import file_sha256
from scripts.repackage_dataset.procedural.suites import VISGYM


def checkpoints(tmp_path):
    roots = {}
    features = Features(dict(images=Sequence(Image(decode=False)),image_group_id=Value('string'),qa=[dict(
        inputs=Value('string'),choices_list=Sequence(Value('string')),labels=Value('int64'),metadata=Value('string'))]))
    for index,family in enumerate(FAMILIES):
        root = tmp_path/family; directory = root/'chunks/00000/build/source-release'
        directory.mkdir(parents=True); roots[family] = str(root)
        for split,count in [('train',2),('validation',1)]:
            rows=[]
            for i in range(count):
                image=io.BytesIO();PILImage.new('RGB',(32,32),(index*50,i*50,100 if split=='train' else 200)).save(image,format='PNG')
                rows.append(dict(images=[dict(bytes=image.getvalue(),path=None)],image_group_id=f'{family}:{split}:{i}',qa=[dict(
                    inputs='Which choice is correct?',choices_list=['correct','incorrect'],labels=0,
                    metadata=json.dumps(dict(verifier_scores=[1,0],seed=i)))]))
            Dataset.from_list(rows,features=features).to_parquet(str(directory/(split+'.parquet')))
        report=dict(revision=VISGYM[1],parquet_sha256={p.name:file_sha256(p) for p in directory.glob('*.parquet')})
        (directory/'provenance.json').write_text(json.dumps(report))
    path=tmp_path/'roots.json';path.write_text(json.dumps(roots))
    return path,roots


def test_one_balanced_mirror_preserves_choices_images_and_split_groups(tmp_path,monkeypatch):
    path,roots=checkpoints(tmp_path);monkeypatch.chdir(tmp_path)
    ds=visgym_balanced(max_rows=8,max_rows_eval=4,checkpoints=path)
    for split,count in [('train',2),('validation',1)]:
        assert Counter(json.loads(qa['metadata'])['family'] for row in ds[split] for qa in row['qa'])==dict.fromkeys(FAMILIES,count)
        for row in ds[split]:
            assert row['images'][0]['bytes'].startswith(b'\x89PNG')
            assert all(qa['choices_list'][qa['labels']]=='correct' for qa in row['qa'])
    assert not set(ds['train']['image_group_id']) & set(ds['validation']['image_group_id'])
    report=json.loads((tmp_path/'build/visgym-mcq-release/provenance.json').read_text())
    assert report['mirror']=='tasksource/visgym-mcq'
    assert report['configuration']['shuffle']
    broken=next((tmp_path/'maze').glob('chunks/*/build/*/train.parquet'))
    broken.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'):
        family_selection(tmp_path/'maze',2,1)


def test_balanced_limits_are_explicit():
    with pytest.raises(ValueError,match='multiples of four'):
        visgym_balanced(max_rows=7,max_rows_eval=4)
