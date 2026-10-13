import hashlib
import io
import json
import zipfile

from datasets import Dataset, DatasetDict
from PIL import Image
import pytest

from scripts.repackage_dataset.games import gameqa_record, valen_records, _package
from tasksource.preprocess import VisualClassification, VisualMultipleChoice
from tasksource.vision_tasks import grouped_qa_rows, grouped_mc_rows
from tasksource.recast import recast_jev


def game(options=None, answer='2'):
    return dict(data_id='game-1',game_name='puzzle',question_id=0,
        question='Count the blocks.\nOptions:\n1: 7\n2: 2\n3: 5',options=options or ['7','2','5'],answer=answer)


def test_gameqa_numeric_answers_are_native_indices_not_option_text():
    row=game(answer='1');qa,reason=gameqa_record(row,0)
    assert reason is None and qa['choices_list'][qa['labels']]=='7'
    assert qa['inputs']=='Count the blocks.'
    assert json.loads(qa['metadata'])['source_question']==row['question']


def test_gameqa_strips_letters_and_remaps_equivalent_duplicate_options():
    qa,reason=gameqa_record(game(['A. red','B. blue','C. red'],answer='C'),1)
    assert reason is None and qa['choices_list']==['red','blue'] and qa['labels']==0
    assert gameqa_record(game(['A. red','C. blue']),1)[1]=='misaligned_source_options'


def test_gameqa_retains_literal_gold_and_excludes_unresolved_formats():
    row=game(['red','blue'],answer='blue');row['question']='What color is the ball?'
    qa,reason=gameqa_record(row,0)
    assert reason is None and qa['choices_list'][qa['labels']]=='blue'
    row['answer']='unknown'
    assert gameqa_record(row,0)[1]=='unresolved_answer'
    row['options']=[]
    assert gameqa_record(row,0)[1]=='no_native_multiple_choice'


def valen():
    return dict(group_id='layout-1',assets=[dict(path='assets/a.png')],meta=dict(record_id='state-1',solution_actions=['up'],language_bucket='en'),
        request=dict(state=dict(messages=[dict(role='user',content=[dict(type='text',text='Push boxes onto goals.'),
            dict(type='image_url',image_url=dict(url='assets/a.png'))])]),questions=dict(next_action=dict(
                type='choice',criteria={a:a.title() for a in ('left','up','down','right')}))),
        targets=dict(next_action=dict(probabilities=dict(up=.5,left=.5,down=0.,right=0.))))


def test_valen_keeps_all_optimal_actions_and_hidden_evidence_out_of_input():
    rows=list(valen_records(valen()))
    by_action={json.loads(r['metadata'])['action']:r for r in rows}
    assert {a for a,r in by_action.items() if r['labels']=='yes'}=={'up','left'}
    assert all('solution_actions' not in r['inputs'] and '0.5' not in r['inputs'] for r in rows)
    assert json.loads(rows[0]['metadata'])['source_metadata']['solution_actions']==['up']
    assert {json.loads(r['metadata'])['split_group_id'] for r in rows}=={'layout-1'}


def test_valen_rejects_mismatched_image_reference():
    row=valen();row['assets'][0]['path']='wrong.png'
    with pytest.raises(AssertionError):list(valen_records(row))


def png(color):
    output=io.BytesIO();Image.new('RGB',(4,4),color).save(output,format='PNG');return output.getvalue()


def test_packaging_preserves_bytes_and_excludes_heldout_images(tmp_path,monkeypatch):
    from scripts.repackage_dataset import games
    monkeypatch.chdir(tmp_path)
    images={'same.png':png('red'),'train.png':png('blue')}
    archives={}
    for split in ('test','train'):
        path=tmp_path/(split+'.zip')
        with zipfile.ZipFile(path,'w') as archive:
            for name,data in images.items():archive.writestr(name,data)
        archives[split]=str(path)
    qa,_=gameqa_record(game(),0)
    groups=dict(test={'same.png':[qa]},train={'same.png':[qa.copy()],'train.png':[qa.copy()]})
    def materialize(data,*args):
        result=DatasetDict({s:Dataset.from_list(list(rows),features=rows.features) for s,rows in data.items()})
        out=tmp_path/'build'/'test-release';out.mkdir(parents=True)
        (out/'provenance.json').write_text('{}')
        return result
    monkeypatch.setattr(games,'_materialize',materialize)
    excluded=[]
    result=_package('test','source','revision',groups,archives,{},excluded,True,gameqa_record,'mit')
    assert len(result['train'])==len(result['test'])==1
    assert result['train'].cast_column('images',result['train'].features['images'])[0]['images'][0]['bytes']==images['train.png']
    assert len(excluded)==1 and excluded[0]['reason']=='heldout_image_or_layout'
    assert result['test'][0]['image_group_id']==hashlib.sha256(images['same.png']).hexdigest()
    canonical=VisualMultipleChoice(choices_list='choices_list')(grouped_mc_rows(result))
    recast=recast_jev(canonical,task='gameqa/mcq')['train'][0]
    assert recast['criteria'][recast['label']]=='2' and 'Options:' not in recast['state']
    assert recast['images'][0].size==(4,4)


def test_valen_binary_recast_keeps_ties_and_images():
    image=png('red');groups=DatasetDict(train=Dataset.from_list([dict(images=[dict(bytes=image,path=None)],
        image_group_id='image',qa=list(valen_records(valen())))]))
    canonical=VisualClassification(label_values={'no':'not an optimal action','yes':'optimal action'})(grouped_qa_rows(groups))
    rows=recast_jev(canonical,task='valen-game/optimal-action')['train']
    assert sum(r['criteria'][r['label']]=='optimal action' for r in rows)==2
    assert all(r['images'][0].size==(4,4) for r in rows)


def test_gameqa_removes_only_known_answer_format_suffixes():
    row=game();row['question']='How many enemies? Give the number of the correct option.\nOptions:\n1: 7\n2: 2\n3: 5'
    qa,reason=gameqa_record(row,0)
    assert reason is None and qa['inputs']=='How many enemies?'


def test_game_catalogs_are_isolated_and_original_sources_can_be_excluded():
    from tasksource import list_tasks
    tasks={'gameqa/mcq','valen-game/optimal-action'}
    assert tasks <= set(list_tasks(vision=True).id)
    assert not tasks & set(list_tasks().id)
    assert 'valen-game/optimal-action' not in set(list_tasks(vision=True,excluded_sources=['Valen-Team/Valen-Eval-Game']).id)


@pytest.mark.parametrize('task',['gameqa/mcq','valen-game/optimal-action'])
def test_shared_loader_jev_preserves_native_gold_and_images(task,monkeypatch):
    from tasksource import access
    image=png('red')
    qas=[gameqa_record(game(answer='1'),0)[0]] if task=='gameqa/mcq' else list(valen_records(valen()))
    data=DatasetDict(train=Dataset.from_list([dict(images=[dict(bytes=image,path=None)],image_group_id='image',qa=qas)]))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:data)
    rows=access.load_task(task,vision=True,recast='jev')['train']
    assert all(r['images'][0].size==(4,4) for r in rows)
    if task=='gameqa/mcq':assert rows[0]['criteria'][rows[0]['label']]=='7'
    else:assert sum(r['criteria'][r['label']]=='Yes' for r in rows)==2


def test_gameqa_question_identity_includes_native_record():
    from tasksource.vision_tasks import gameqa_metadata
    first, _ = gameqa_record(game(), 1)
    second, _ = gameqa_record(game(), 2)
    assert gameqa_metadata(first)['question_id'] != gameqa_metadata(second)['question_id']
    assert gameqa_metadata(first)['source_question_id'] == game()['question_id']


def test_valen_language_filter_precedes_sampling_and_preserves_splits():
    from tasksource.vision_tasks import valen_english_rows
    en, zh = valen(), valen()
    zh['meta']['language_bucket'] = 'zh'
    data = DatasetDict({split: Dataset.from_list([dict(images=[dict(bytes=png('red'), path=None)],
        image_group_id='image', qa=list(valen_records(en)) + list(valen_records(zh)))])
        for split in ('train', 'test')})
    selected = valen_english_rows(data)
    assert set(selected) == {'train', 'test'}
    assert all(len(rows) == 4 for rows in selected.values())
    assert all(json.loads(row['metadata'])['source_metadata']['language_bucket'] == 'en'
               for rows in selected.values() for row in rows)
