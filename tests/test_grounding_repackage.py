"""Native boxes, canonical choices and stable source sampling agree."""
import json

from scripts.repackage_dataset.grounding import grounding_qa, ordered
from tasksource.grounding import coordinate_text


def test_native_target_required_and_choices_aligned():
    boxes = [[i*.075,.2,i*.075+.04,.4] for i in range(12)]
    metadata = dict(source_row='screen:7', image_group_id='screen')
    excluded = []
    qa = grounding_qa('Find the blue widget', boxes[7], boxes, metadata, 'fixture', excluded)
    info = json.loads(qa['metadata'])
    assert len(qa['choices_list']) == 8
    assert qa['choices_list'] == list(map(coordinate_text, info['candidate_boxes']))
    assert info['candidate_boxes'][qa['labels']] == boxes[7]
    assert not excluded
    assert grounding_qa('Find widget', boxes[7], boxes[:7], metadata, 'fixture', excluded) is None
    assert excluded[-1]['reason'] == 'target_not_in_native_candidates'


def test_distinct_native_distractor_and_instruction_required():
    box = [0, 0, .1, .1]
    meta = dict(source_row='x', image_group_id='x')
    excluded = []
    assert grounding_qa('Find widget', box, [box, box], meta, 'fixture', excluded) is None
    assert excluded[-1]['reason'] == 'no_distinct_native_distractor'
    assert grounding_qa(' ', box, [box], meta, 'fixture', excluded) is None
    assert excluded[-1]['reason'] == 'empty_instruction'
    assert ordered(['c','a','b'], 1) == ordered(['b','c','a'], 1)


def test_callable_materialization_uses_canonical_grouped_features(tmp_path, monkeypatch):
    import io
    from PIL import Image as PILImage
    from scripts.repackage_dataset.grounding import materialize
    monkeypatch.chdir(tmp_path)
    buffer = io.BytesIO();PILImage.new('RGB', (16, 16), 'white').save(buffer, format='PNG')
    qa = grounding_qa('Left square', [0,0,.3,.3], [[0,0,.3,.3],[.6,.6,1,1]],
                      dict(source_row='x',image_group_id='x'), 'fixture', [])
    def rows():
        yield dict(images=[dict(bytes=buffer.getvalue(),path=None)],image_group_id='x',qa=[qa])
    result = materialize({'train':rows}, 'fixture', [('fixture/source','pinned')], [],
                         grounding_qa, 'https://example.org/terms', 100000, 0)
    report = json.loads((tmp_path/'build/fixture-release/provenance.json').read_text())
    assert report['splits']['train'] == dict(rows=1, questions=1)
    assert report['max_train_questions'] == 100000
    assert len(report['parquet_sha256']['train.parquet']) == 64
    assert result['train'][0]['qa'][0]['labels'] == qa['labels']


def test_refcoco_reserves_heldout_bytes_under_different_image_ids(tmp_path, monkeypatch):
    import hashlib
    import io
    import zipfile
    from PIL import Image as PILImage
    from scripts.repackage_dataset import grounding
    monkeypatch.chdir(tmp_path)
    directory = tmp_path/'build/grounding-assets';directory.mkdir(parents=True)
    def image(color):
        output=io.BytesIO();PILImage.new('RGB',(32,32),color).save(output,format='PNG');return output.getvalue()
    photographs={1:image('red'),2:image('red'),3:image('blue'),4:image('green')}
    images=[dict(id=i,file_name=f'{i}.png',width=32,height=32,license=1) for i in photographs]
    annotations=[dict(id=i*10+j,image_id=i,bbox=box,iscrowd=0) for i in photographs
                 for j,box in enumerate(([0,0,8,8],[16,16,8,8]))]
    source=dict(images=images,annotations=annotations,licenses=[dict(id=1,name='Attribution',url='https://creativecommons.org/licenses/by/2.0/')])
    path=directory/'annotations_trainval2014.zip'
    with zipfile.ZipFile(path,'w') as archive:archive.writestr('annotations/instances_train2014.json',json.dumps(source))
    with zipfile.ZipFile(directory/'train2014.zip','w') as archive:
        for i,encoded in photographs.items():archive.writestr(f'train2014/{i}.png',encoded)
    def row(i):
        return dict(image_id=i,ann_id=i*10,ref_id=i,sentences=[dict(sent_id=i,raw='The upper left object')])
    monkeypatch.setattr(grounding,'COCO_ANNOTATIONS_SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(grounding,'REFCOCO',{'fixture':('fixture/source','pin')})
    monkeypatch.setattr(grounding,'_parquet_source',lambda *a:dict(train=[row(1),row(3)],validation=[row(4)],test=[row(2)]))
    result=grounding.refcoco_grounding()
    assert list(result['train']['image_group_id']) == ['coco:3']
    assert list(result['test']['image_group_id']) == ['coco:2']
    assert list(result['validation']['image_group_id']) == ['coco:4']
