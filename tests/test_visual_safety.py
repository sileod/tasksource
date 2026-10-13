import json
import pytest
from datasets import Dataset,DatasetDict
from scripts.repackage_dataset.visual_safety import t2i_record,TOXICITY_POLICY


def source(category='violence',answer='unsafe\nviolence',real=False):
    return dict(id=7,image_category=category,text_category='violence',base_category='toxicity',
        caption='A generation prompt that must never become evidence.',real=real,
        conversations=[{'from':'user','value':''},{'from':'assistant','value':answer}],image=['image.png'])


@pytest.mark.parametrize('category,answer',[('safe','safe'),('violence','unsafe\nviolence'),
    ('disturbing and violence','unsafe\nviolence and disturbing')])
def test_t2i_preserves_native_binary_labels_without_caption_leakage(category,answer):
    row=source(category=category,answer=answer);qa,reason=t2i_record(row,'train')
    assert reason is None and qa['labels']==answer.splitlines()[0]
    assert qa['inputs']==TOXICITY_POLICY and row['caption'] not in qa['inputs']
    assert json.loads(qa['metadata'])['source_answer']==answer


def test_t2i_excludes_unreviewed_domains_without_inventing_safe_labels():
    for category in ('public figures','public figures and violence','intellectual property violation'):
        qa,reason=t2i_record(source(category=category,answer='unsafe\n'+category),'train')
        assert qa is None and reason=='privacy_outside_toxicity_policy'
    assert t2i_record(source(real=True),'train')[1]=='real_image_outside_reviewed_scope'


def test_t2i_rejects_mismatched_native_answer():
    with pytest.raises(ValueError,match='answer/category mismatch'):
        t2i_record(source(answer='unsafe\nsexual'),'train')
