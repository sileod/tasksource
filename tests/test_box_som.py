"""Coordinate selection keeps native geometry aligned with SoM and Jev choices."""
import io
import json

import pytest
from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from tasksource.grounding import augment_grounding, coordinate_choices, coordinate_text
from tasksource.preprocess import VisualMultipleChoice
from tasksource.recast import recast_jev
from tasksource.vision_tasks import box_choice_rows
from PIL import Image as PILImage


def image():
    stream=io.BytesIO();PILImage.new("RGB",(100,80),"gray").save(stream,format="PNG")
    return {"bytes":stream.getvalue(),"path":None}


def test_coordinate_geometry_comes_from_the_selected_accepted_candidate():
    target=[0,0,.1,.1]
    # Same rounded coordinates; only the second box clears the distance threshold.
    near=[.0296,0,.1296,.1];far=[.0304,0,.1304,.1]
    choices,label,boxes=coordinate_choices(target,[near,far],key='rounding',return_geometry=True)
    assert boxes[label]==target and far in boxes and near not in boxes
    assert choices==list(map(coordinate_text,boxes))


@pytest.mark.parametrize('variant',['plain','som+text','som-only'])
def test_sampled_boxes_marks_and_gold_survive_jev(variant):
    boxes=[[i*.075,.2,i*.075+.04,.4] for i in range(12)];gold=boxes[7]
    row=dict(images=[image()],inputs='Find the described widget',choices_list=['unused'],labels=0,
             metadata=json.dumps(dict(id='screen:widget',source_dataset='fixture',image_group_id='screen',
                                      target_bbox=gold,candidate_boxes=boxes)))
    features=Features(dict(images=Sequence(Image(decode=False)),inputs=Value('string'),
        choices_list=Sequence(Value('string')),labels=Value('int64'),metadata=Value('string')))
    raw=DatasetDict(train=Dataset.from_list([row],features=features))
    projected=box_choice_rows(raw,lambda r:(gold,boxes))
    selected=projected['train'][0];meta=json.loads(selected['metadata'])
    assert len(meta['candidate_boxes'])==8
    assert selected['choices_list']==list(map(coordinate_text,meta['candidate_boxes']))
    assert meta['candidate_boxes'][selected['labels']]==gold
    processed=VisualMultipleChoice(choices_list='choices_list',metadata='metadata')(projected)
    augmented=augment_grounding(processed,{variant:1},seed=4)
    result=recast_jev(augmented,task='fixture/box')['train'][0]
    mark=selected['labels']+1
    expected=coordinate_text(gold) if variant=='plain' else f'Mark {mark}'+(
        ': '+coordinate_text(gold) if variant=='som+text' else '')
    assert result['criteria'][result['label']]==expected
    info=json.loads(result['metadata'])
    assert info['augmentation']['candidate_boxes']==meta['candidate_boxes']
    assert 'target_bbox' not in result['state']
