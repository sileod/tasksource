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


def test_t2i_shared_loader_preserves_images_policy_gold_and_splits(monkeypatch):
    import io
    from PIL import Image
    import tasksource.access as access
    from tasksource import load_task, list_tasks
    from tasksource.preprocess import disable_image_decoding
    buffer=io.BytesIO();Image.new('RGB',(4,4),'blue').save(buffer,format='PNG')
    image={'bytes':buffer.getvalue(),'path':None}
    qa,_=t2i_record(source(),'train')
    native=DatasetDict({split:Dataset.from_list([{'images':[image],
        'image_group_id':'same-image','qa':[qa]}]) for split in ('train','test')})
    monkeypatch.setattr(access,'load_dataset',lambda *args,**kwargs:native)
    result=load_task('t2isafety/toxicity',vision=True,recast='jev')
    assert set(result)=={'train','test'}
    for split in result:
        row=disable_image_decoding(result)[split][0]
        assert row['images']==[image]
        assert row['answer']=='unsafe'==row['criteria'][row['label']]
        assert TOXICITY_POLICY in row['state']
        assert source()['caption'] not in row['state']
    assert 't2isafety/toxicity' not in set(list_tasks().id)
