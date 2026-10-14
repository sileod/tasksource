import io
import json

import pytest
from PIL import Image as PILImage

from scripts.repackage_dataset.procedural.common import seed_for, verified_choices, write_static
from scripts.repackage_dataset.procedural.trace import trace_question
from scripts.repackage_dataset.procedural.tron import clean_question, native_mc,tron_question
from tasksource import recast_jev
from tasksource.preprocess import VisualMultipleChoice, disable_image_decoding
from tasksource.vision_tasks import grouped_mc_rows


def test_distractors_reject_numeric_equivalents_and_keep_donor_alignment():
    verify=lambda x:int(abs(float(x)-1)<0.1)
    donors=[('1.0','equivalent'),('1.05','within_tolerance'),('2','two'),('3','three'),('4','four')]
    first=verified_choices('1',donors,verify,42)
    assert first==verified_choices('1',donors,verify,42)
    choices,label,metadata=first
    assert set(choices)=={'1','2','3','4'} and choices[label]=='1'
    assert metadata['verifier_scores']==[int(i==label) for i in range(4)]
    assert metadata['donor_ids']==[None if c=='1' else {'2':'two','3':'three','4':'four'}[c] for c in choices]
    assert seed_for('train',42)!=seed_for('validation',42)


def test_no_fake_padding_and_bad_gold_fails():
    choices,label,_=verified_choices('yes',[('no','other'),('NO','duplicate')],lambda a:int(a=='yes'),0)
    assert len(choices)==2 and choices[label]=='yes'
    with pytest.raises(ValueError,match='No verifier-rejected'):
        verified_choices('yes',[('yes','same')],lambda a:int(a=='yes'),0)
    with pytest.raises(ValueError,match='rejects its own gold'):
        verified_choices('yes',[('no','other')],lambda a:0,0)


def test_native_mc_preserves_content_and_rejects_opaque_or_multiple_golds():
    question='Which region is wet?\nA. Water (W)\nB. Air (A)\nC. Undetermined\nAnswer with a single letter A, B, or C.'
    stem,choices,label,metadata=native_mc(question,'B',lambda a:int(a=='B'))
    assert stem=='Which region is wet?' and choices==['Water (W)','Air (A)','Undetermined'] and label==1
    assert metadata['verifier_scores']==[0,1,0]
    inline='What is the total?\n(A)350 (B)349 (C)385 (D)326\nPick the correct letter (A-D). Reply with the letter inside <answer>X</answer>.'
    stem,choices,label,_=native_mc(inline,'A',lambda a:int(a=='A'))
    assert stem=='What is the total?' and choices==['350','349','385','326'] and label==0
    inline=inline.replace('Pick the correct letter (A-D). Reply with the letter inside',
        'Select the correct option (A-D) and place the single letter in')
    assert native_mc(inline,'A',lambda a:int(a=='A'))[1]==['350','349','385','326']
    with pytest.raises(ValueError,match='without explicit'):
        native_mc('Look at the figure.','A',lambda a:int(a=='A'))
    with pytest.raises(ValueError,match='Ambiguous'):
        native_mc(question,'B',lambda a:1)
    assert clean_question('What is the area? Numeric answer in <answer>...</answer>.')=='What is the area?'


def test_tron_rounded_measurements_use_closest_offered_value():
    question='Find half of the shown angle. Round to 2 decimals.'
    assert 'Round to 2 decimals' not in tron_question(question)
    assert 'closest offered value' in tron_question(question)
    assert tron_question('How many circles?')=='How many circles?'
    # The source displays 51.3 degrees but derives its gold from hidden precision.
    choices=[25.67,5.33,3,9]
    assert min(range(len(choices)),key=lambda i:abs(choices[i]-51.3/2))==0


@pytest.mark.parametrize('prompt',[
    '<image>What is the area?\nFormat for the "answer" field: set "answer" to the integer area in square units.\nExample JSON:\n{"answer":12}',
    '<image>What is the area?\nReturn a JSON object with "answer" only.\nRequired answer format: set "answer" to the integer area in square units.\nExample JSON:\n{"answer":12}',
    '<image>What is the area?\nFinal answer format: set "answer" to the integer area in square units.\nExample JSON:\n{"answer":12}',
    'What is the area?\nReturn a JSON object.\nAnswer field: set "answer" to the integer area in square units.\nExample JSON:\n{"answer":12}',
    'What is the area?\nReturn only a JSON object containing "answer".\nAnswer field: set "answer" to the integer area in square units.',
    'What is the area?\nUse JSON only.\nAnswer field: set "answer" to the integer area in square units.',
    'What is the area?\nRespond with JSON only.\nAnswer field: set "answer" to the integer area in square units.',
])
def test_trace_prompt_keeps_units_without_image_tokens_or_example_answers(prompt):
    assert trace_question(prompt)=='What is the area?\nChoose the integer area in square units.'
    assert trace_question(prompt,'toggle_grid').startswith('Pressing a switch flips that cell and its orthogonal neighbors')
    assert 'Json' in trace_question(prompt.replace('What is the area?','Find the Json series in the chart.'))
    with pytest.raises(ValueError,match='Unknown TRACE'):
        trace_question('Ambiguous output contract.')


def test_static_image_groups_splits_and_jev_permutation(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    def row(identity,color):
        buffer=io.BytesIO();PILImage.new('RGB',(3,3),color).save(buffer,format='PNG')
        return dict(id=identity,image=buffer.getvalue(),inputs='Which value fits the diagram?',
            choices_list=['1','3','5'],labels=1,metadata=dict(task='native-task',seed=seed_for(identity)))
    train=row('t','red');heldout=row('v','blue')
    ds=write_static('test',dict(train=[train,{**train,'id':'same-image-question'},heldout],validation=[heldout]),
        dict(source='source',revision='pin',license='other',license_url='terms'),write_static)
    assert len(ds['train'])==len(ds['validation'])==1 and len(ds['train'][0]['qa'])==2
    report=json.loads((tmp_path/'build/test-release/provenance.json').read_text())
    assert report['image_overlap_excluded']==1
    from scripts.upload_repackaged import load_prepared
    path=tmp_path/'build/test-release'
    assert len(load_prepared(path,'tasksource/test')['train'])==1
    with pytest.raises(ValueError,match='another mirror'):
        load_prepared(path,'tasksource/wrong')
    train_file=path/'train.parquet'
    train_file.write_bytes(train_file.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum mismatch'):
        load_prepared(path,'tasksource/test')
    task=VisualMultipleChoice(choices_list='choices_list',metadata='metadata',pre_process=grouped_mc_rows)
    result=disable_image_decoding(recast_jev(task(ds),task='test'))
    assert len(result['train'])==2
    for r in result['train']:
        assert r['answer']=='3'==r['criteria'][r['label']]
        assert r['images'][0]['bytes']==train['image']
        assert json.loads(r['metadata'])['task']=='native-task'


def test_scale_merges_image_groups_rejects_overlap_and_caps_distinct_questions(tmp_path):
    from collections import Counter
    from datasets import Dataset
    from scripts.repackage_dataset.procedural.scale import collect,examples
    def row(image,question):
        return dict(images=[dict(bytes=image.encode(),path=None)],image_group_id=image,qa=[dict(
            inputs=question,choices_list=['wrong','gold'],labels=1,
            metadata=json.dumps(dict(verifier_scores=[0,1])))])
    first=tmp_path/'first';second=tmp_path/'second';first.mkdir();second.mkdir()
    Dataset.from_list([row('heldout','v')]).to_parquet(first/'validation.parquet')
    Dataset.from_list([row('one','q')]).to_parquet(first/'train.parquet')
    Dataset.from_list([row('heldout','leak'),row('one','q'),row('one','another'),row('two','over cap')]).to_parquet(second/'train.parquet')
    groups={'validation':{},'train':{}};fingerprints={s:set() for s in groups};excluded=Counter()
    for directory in (first,second):collect(directory,groups,fingerprints,2,excluded)
    merged=list(examples([first,second],'train',groups['train']))
    assert len(merged)==1 and len(merged[0]['qa'])==2
    assert [q['inputs'] for q in merged[0]['qa']]==['q','another']
    assert merged[0]['images'][0]['bytes']==b'one'
    assert excluded==dict(image_in_validation=1,duplicate_question=1,train_cap=1)
    from datasets import IterableDataset
    streamed=IterableDataset.from_generator(examples,gen_kwargs=dict(
        directories=(str(first),str(second)),split='train',groups=groups['train']))
    assert list(streamed)==merged


def test_fresh_trace_split_identity_ignores_png_encoding():
    from scripts.repackage_dataset.procedural.trace_generate import pixel_hash
    image=PILImage.new('RGB',(4,4),'red');first=io.BytesIO();second=io.BytesIO()
    image.save(first,format='PNG',compress_level=0);image.save(second,format='PNG',compress_level=9)
    assert first.getvalue()!=second.getvalue()
    assert pixel_hash(first.getvalue())==pixel_hash(second.getvalue())
    other=io.BytesIO();PILImage.new('RGB',(2,8),'red').save(other,format='PNG')
    assert pixel_hash(other.getvalue())!=pixel_hash(first.getvalue())
