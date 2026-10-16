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
