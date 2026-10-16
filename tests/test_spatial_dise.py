"""Native panel identity, gold indices and conservative source split isolation."""
import io
import json

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from PIL import Image as PILImage
from tasksource import access, load_task
from scripts.repackage_dataset.spatial_dise import select_rows, source_group


def row(number, category='3D Folding', difficulty='easy', answer='D', options='A, B, C, D'):
    return dict(category=category, source='synthetic', difficulty=difficulty, question='Which cube folds from this net?',
                image=f'images/synthetic/merged/{category}/{difficulty}/merged_question_{number:04d}.png',
                answer=answer, options=options)


def test_native_holdouts_and_benchmark_reserve_all_difficulty_variants():
    splits=dict(benchmark=[row(1)], test=[row(1),row(2)], validation=[row(2,'3D Folding','hard'),row(3)],
                train=[row(3,'3D Folding','medium'),row(4),row(5,'3D Projection'),row(6,answer='E')])
    selected,excluded=select_rows(splits)
    assert {s:[source_group(r) for r in rs] for s,rs in selected.items()}=={
        'test':['3D Folding:0002'],'validation':['3D Folding:0003'],'train':['3D Folding:0004']}
    assert sorted(r['reason'] for r in excluded)==[
        'heldout_source_group','heldout_source_group','heldout_source_group','invalid_choices','unaudited_category']


def test_duplicate_or_noncontiguous_panel_references_are_rejected():
    selected,_=select_rows(dict(benchmark=[],test=[],validation=[],train=[
        row(1,options='A, A, C, D'),row(2,options='A, C, D'),row(3,options='A, B',answer='B')]))
    assert [r['answer'] for r in selected['train']]==['B']


def test_panel_gold_and_original_bytes_survive_jev_permutation(monkeypatch):
    buffer=io.BytesIO();PILImage.new('RGB',(2,2),'red').save(buffer,format='PNG');payload=buffer.getvalue()
    raw=DatasetDict(train=Dataset.from_list([
        dict(images=[dict(bytes=payload,path=None)],image_group_id='same-screen',qa=[dict(
            inputs=f'Question {i}: choose the matching image panel.',choices_list=['Panel '+v for v in 'ABCD'],
            labels=i%4,metadata=json.dumps(dict(native_answer='ABCD'[i%4],id=str(i))))]) for i in range(12)
    ],features=Features(dict(images=Sequence(Image(decode=False)),image_group_id=Value('string'),qa=[dict(
        inputs=Value('string'),choices_list=Sequence(Value('string')),labels=Value('int64'),metadata=Value('string'))]))))
    monkeypatch.setattr(access,'load_dataset',lambda *a,**k:raw)
    result=load_task('spatial-dise/mcq',vision=True,recast='jev')['train']
    assert len(result)==12
    moved=False
    for r in result.cast_column('images',Sequence(Image(decode=False))):
        meta=json.loads(r['metadata']);assert r['criteria'][r['label']]=='Panel '+meta['native_answer']
        assert r['images'][0]['bytes']==payload
        assert 'native_answer' not in r['state'] and meta['image_group_id']=='same-screen'
        moved |= r['criteria']!=['Panel '+v for v in 'ABCD']
    assert moved
