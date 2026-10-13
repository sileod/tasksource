import io
import json
import zipfile

import pytest
from datasets import Dataset,DatasetDict,Features,Image,Value
from PIL import Image as PILImage

from tasksource import load_task
from tasksource.preprocess import disable_image_decoding
from scripts.repackage_dataset.memes import memeqa_record
from scripts.repackage_dataset.games import _package


def png(color):
    data=io.BytesIO();PILImage.new('RGB',(3,3),color).save(data,format='PNG')
    return {'bytes':data.getvalue(),'path':None}


def test_native_mc_gold_and_multiselect_exclusion():
    row=dict(id='q',img='m.png',question='Who is praised?',options=['A','B','C'],
             general_type='single',specific_type='Intent Identification',answer_key=2)
    qa,reason=memeqa_record(row,'train')
    assert reason is None and qa['labels']==2 and qa['choices_list']==row['options']
    assert 'answer' not in qa['inputs']
    assert memeqa_record({**row,'general_type':'multi','answer_key':[1,0,1]},'train')[1]=='native_multiselect_outside_mc_view'
    assert memeqa_record({**row,'options':['A','a']},'train')[1]=='duplicate_choices'
    with pytest.raises(ValueError):memeqa_record({**row,'answer_key':3},'train')


def test_archive_packager_preserves_three_splits_and_heldout_images(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    path=tmp_path/'images.zip'
    with zipfile.ZipFile(path,'w') as z:
        for color in ('red','blue','green'):z.writestr(color+'.png',png(color)['bytes'])
    qa=lambda i:dict(inputs='Question?',choices_list=['wrong','gold'],labels=1,metadata=json.dumps({'id':i}))
    groups={'train':{'red.png':[qa('r')],'blue.png':[qa('copy')]},
            'validation':{'blue.png':[qa('b')]},'test':{'green.png':[qa('g')]}}
    ds=_package('test','source','revision',groups,{s:str(path) for s in groups},{},[],True,png,'other')
    assert {s:len(d) for s,d in ds.items()}=={'train':1,'validation':1,'test':1}
    assert disable_image_decoding(ds)['train'][0]['images']==[png('red')]
    assert ds['train'][0]['qa'][0]['labels']==1


@pytest.mark.parametrize('task,gold',[('sentiment','happiness'),('intention','entertaining'),
                                     ('offensiveness','non-offensive'),('metaphor','literal')])
def test_met_meme_native_labels_english_and_image_holdouts(monkeypatch,task,gold):
    import tasksource.access as access
    row={'images_name':'English/r.jpg','text':'Native meme text','image':png('red'),
         'sentiment category':'1(happiness)','intention detection':'3(entertaining)',
         'offensiveness detection':'0(non-offensive)','metaphor occurrence':'0'}
    features=Features({k:Image(decode=False) if k=='image' else Value('string') for k in row})
    native=DatasetDict(train=Dataset.from_list([row,{**row,'images_name':'English/copy.jpg','image':png('blue')},
        {**row,'images_name':'Chinese/x.jpg'}],features=features),
        validation=Dataset.from_list([{**row,'image':png('blue')}],features=features))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**kw:native)
    result=load_task('met-meme/'+task,vision=True,recast='jev')
    assert set(result)=={'train','validation'} and len(result['train'])==1
    recast=disable_image_decoding(result)['train'][0]
    assert recast['images']==[png('red')]
    assert recast['answer']==gold==recast['criteria'][recast['label']]
    assert 'Native meme text' in recast['state']


@pytest.mark.parametrize('streaming',[False,True])
def test_met_conflicting_images_excluded_without_resolving_labels(streaming):
    from tasksource.vision_tasks import met_meme_english_rows
    row={'images_name':'English/a.jpg','image':png('red'),'sentiment category':'1(happiness)',
         'intention detection':'3(entertaining)','offensiveness detection':'0(non-offensive)','metaphor occurrence':'0'}
    features=Features({k:Image(decode=False) if k=='image' else Value('string') for k in row})
    native=DatasetDict(train=Dataset.from_list([row,{**row,'image':png('blue')}],features=features),
        test=Dataset.from_list([row,{**row,'metaphor occurrence':'1'},{**row,'image':png('green')}],features=features))
    if streaming:
        from datasets import IterableDatasetDict
        native=IterableDatasetDict({s:rows.to_iterable_dataset() for s,rows in native.items()})
    result={s:list(rows) for s,rows in met_meme_english_rows(native).items()}
    assert len(result['test'])==len(result['train'])==1
    assert result['train'][0]['image']==png('blue')
